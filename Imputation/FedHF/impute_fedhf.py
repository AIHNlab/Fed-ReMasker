import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm

from FedHF.model import FedHFImputer
from aggregation import fedavg
from data_utils import get_hp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hocine et al. (2025), except where noted.
_DEFAULTS = dict(
    global_rounds=60,  # paper: 20 -- raised to match the other methods, so the
                       # comparison is at an equal round budget rather than
                       # crediting this method with fewer rounds
    local_epochs=2,
    batch_size=64,
    d_model=128,
    n_layers=3,
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    block_frac=0.25,   # base corruption fraction (overridden by schedule each round)
    topk=8,            # top-k correlated neighbours per feature in graph
    min_abs_corr=0.05,  # minimum absolute correlation to include an edge
    seed=0,
)


def _get(args, key):
    return get_hp(args, key, _DEFAULTS)


def _corruption_schedule(rnd: int) -> float:
    """Match the paper's schedule: start hard, ease as training progresses."""
    if rnd <= 5:
        return 0.7
    elif rnd <= 12:
        return 0.5
    else:
        return 0.3


# ---------------------------------------------------------------------------
# Graph construction (federated: pooled sums, no raw data shared)
# ---------------------------------------------------------------------------

def _build_feature_graph(client_arrays, topk, min_abs_corr):
    """
    Compute pairwise Pearson correlation across all clients using only
    pooled sufficient statistics (sums and counts) -- no raw rows shared.
    Then build a directed top-k graph weighted by |correlation|.
    """
    F = client_arrays[0].shape[1]
    sum_x = np.zeros(F, dtype=np.float64)
    sum_x2 = np.zeros(F, dtype=np.float64)
    cnt_x = np.zeros(F, dtype=np.float64)
    sum_xy = np.zeros((F, F), dtype=np.float64)
    cnt_xy = np.zeros((F, F), dtype=np.float64)

    for X in client_arrays:
        m = ~np.isnan(X)
        X0 = np.nan_to_num(X, nan=0.0)
        sum_x += (X0 * m).sum(axis=0)
        sum_x2 += ((X0 ** 2) * m).sum(axis=0)
        cnt_x += m.sum(axis=0)
        cnt_xy += m.T @ m
        sum_xy += X0.T @ X0

    mean = sum_x / np.maximum(cnt_x, 1.0)
    var = sum_x2 / np.maximum(cnt_x, 1.0) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    exy = sum_xy / np.maximum(cnt_xy, 1.0)
    cov = exy - mean[:, None] * mean[None, :]
    corr = cov / (std[:, None] * std[None, :])
    corr[np.isnan(corr)] = 0.0
    np.fill_diagonal(corr, 0.0)

    abs_corr = np.abs(corr)
    src, dst, wts = [], [], []
    for i in range(F):
        order = np.argsort(-abs_corr[i])
        k = 0
        for j in order:
            if j == i:
                continue
            if abs_corr[i, j] < min_abs_corr:
                break
            src.append(i)
            dst.append(j)
            wts.append(float(abs_corr[i, j]))
            k += 1
            if k >= topk:
                break

    if not src:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
        edge_weight = torch.zeros(0, dtype=torch.float32, device=DEVICE)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=DEVICE)
        edge_weight = torch.tensor(wts, dtype=torch.float32, device=DEVICE)

    return edge_index, edge_weight


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def _make_loader(X_np, obs_np, batch_size):
    x = torch.tensor(X_np, dtype=torch.float32)
    obs = torch.tensor(obs_np, dtype=torch.float32)
    avail = torch.ones_like(obs)   # all features in schema for every client
    ds = TensorDataset(x, obs, avail)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


def _block_corrupt(obs, avail, block_frac, rng):
    """Randomly mask block_frac of the observed features (whole columns per batch)."""
    B, F = obs.shape
    obs_any = (obs > 0).any(dim=0)
    avail_f = (avail[0] > 0)
    candidates = (avail_f & obs_any).nonzero(as_tuple=False).flatten()
    if candidates.numel() == 0:
        return torch.zeros(B, F, device=obs.device, dtype=torch.bool)
    m = max(1, int(math.ceil(block_frac * candidates.numel())))
    perm = candidates[torch.randperm(candidates.numel(), generator=rng, device=obs.device)]
    chosen = perm[:m]
    corrupt = torch.zeros(B, F, device=obs.device, dtype=torch.bool)
    corrupt[:, chosen] = True
    return corrupt & (obs > 0)


def _train_local(model, loader, optimizer, block_frac, local_epochs, rng):
    model.to(DEVICE)
    model.train()
    edge_index = model._ei
    edge_weight = model._ew
    final_loss = 0.0

    for _ in range(local_epochs):
        total, n = 0.0, 0
        for x, obs, avail in loader:
            x, obs, avail = x.to(DEVICE), obs.to(DEVICE), avail.to(DEVICE)
            corrupt = _block_corrupt(obs, avail, block_frac, rng)
            if corrupt.sum() == 0:
                continue
            x_in = x.clone()
            x_in[corrupt] = 0.0
            obs_in = obs.clone()
            obs_in[corrupt] = 0.0
            pred = model(x_in, obs_in, avail, edge_index, edge_weight)
            loss = ((pred - x) ** 2)[corrupt].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1
        if n:
            final_loss = total / n

    model.to("cpu")
    return final_loss


# ---------------------------------------------------------------------------
# Imputation at inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _impute(model, X_np, obs_np, batch_size):
    """Run model on full data; return filled array (only missing positions updated)."""
    model.to(DEVICE)
    model.eval()
    edge_index = model._ei
    edge_weight = model._ew

    n, F = X_np.shape
    out = X_np.copy()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.tensor(X_np[start:end], dtype=torch.float32, device=DEVICE)
        obs = torch.tensor(obs_np[start:end], dtype=torch.float32, device=DEVICE)
        avail = torch.ones_like(obs)
        pred = model(x, obs, avail, edge_index, edge_weight)
        out[start:end] = pred.cpu().numpy()

    model.to("cpu")
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def impute_fedhf_file(dfs_dict_norm, args, global_stats=None):
    """
    Federated GNN-based imputation (FedHF style).

    Parameters
    ----------
    dfs_dict_norm : dict {filename: DataFrame}
    args          : argparse.Namespace
    global_stats  : not used (kept for API parity)

    Returns
    -------
    dict {filename: DataFrame}
    """
    from data_utils import log_hps
    log_hps(args, _DEFAULTS, "Fed-HF")

    global_rounds = _get(args, "global_rounds")
    local_epochs = _get(args, "local_epochs")
    batch_size = _get(args, "batch_size")
    d_model = _get(args, "d_model")
    n_layers = _get(args, "n_layers")
    dropout = _get(args, "dropout")
    lr = _get(args, "lr")
    weight_decay = _get(args, "weight_decay")
    topk = _get(args, "topk")
    min_abs_corr = _get(args, "min_abs_corr")
    seed = _get(args, "seed")

    filenames = list(dfs_dict_norm.keys())
    columns = list(list(dfs_dict_norm.values())[0].columns)
    n_features = len(columns)

    # --- numpy arrays and observation masks ---
    client_X = {}
    client_obs = {}
    for fname, df in dfs_dict_norm.items():
        X = df.values.astype(np.float32)
        client_obs[fname] = (~np.isnan(X)).astype(np.float32)
        client_X[fname] = np.nan_to_num(X, nan=0.0)

    # --- build global feature correlation graph (federated) ---
    edge_index, edge_weight = _build_feature_graph(
        [df.values.astype(np.float64) for df in dfs_dict_norm.values()],
        topk=topk, min_abs_corr=min_abs_corr,
    )

    # --- initialise one shared model ---
    torch.manual_seed(seed)
    model = FedHFImputer(n_features=n_features, d_model=d_model,
                         n_layers=n_layers, dropout=dropout)
    model._ei = edge_index
    model._ew = edge_weight

    # one optimizer per client (local state not shared)
    optimizers = {
        fname: torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        for fname in filenames
    }
    loaders = {
        fname: _make_loader(client_X[fname], client_obs[fname], batch_size)
        for fname in filenames
    }

    # --- federated loop ---
    for rnd in trange(1, global_rounds + 1, desc="Fed-HF global round", colour="magenta"):
        tqdm.write(f"  [FedHF] Round {rnd}/{global_rounds}")

        current_block_frac = _corruption_schedule(rnd)
        results = []
        for i, fname in enumerate(filenames):
            rng = torch.Generator(device=DEVICE)
            rng.manual_seed(seed + rnd * 1000 + i)
            _train_local(model, loaders[fname], optimizers[fname],
                         current_block_frac, local_epochs, rng)
            results.append({
                "params": model.state_dict(),
                "num_steps": client_X[fname].shape[0],
                "client_id": fname,
                "round": rnd,
            })

        global_weights = fedavg(results)
        model.load_state_dict(global_weights)

    # --- final imputation ---
    dfs_imputed = {}
    for fname, df in dfs_dict_norm.items():
        X_imp = _impute(model, client_X[fname], client_obs[fname], batch_size)
        df_out = df.copy()
        mask = np.isnan(df.values)
        for j in range(df.shape[1]):
            missing_rows = np.where(mask[:, j])[0]
            if len(missing_rows):
                df_out.iloc[missing_rows, j] = X_imp[missing_rows, j]
        dfs_imputed[fname] = df_out

    return dfs_imputed
