import numpy as np
import torch
from tqdm import trange, tqdm
from Miwae.model import MIWAE
from aggregation import fedavg
from data_utils import get_hp
from torch.nn.utils import parameters_to_vector
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DEFAULTS = dict(
    global_rounds=60,    # CAFE paper (Min et al. 2025) imp_config_tmplate_miwae.yaml
    local_epochs=5,      # CAFE paper imp_config_tmplate_miwae.yaml
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=64,
    K=20,                # CAFE paper miwae.py default
    L=1000,              # CAFE paper miwae.py default
    n_hidden=256,        # Fed-MIWAE paper Table S3
    latent_size=20,      # Fed-MIWAE paper Table S3
    fed_prox_mu=0,       # 0 = plain FedAvg, matching the CAFE reference implementation
                         # (client_vae.py / strategy/fedavg.py: sample-size-weighted
                         # FedAvg, no proximal term). Set >0 to enable FedProx.
    seed=0,
)


def _get(args, key):
    return get_hp(args, key, _DEFAULTS)


def _df_to_arrays(df):
    """DataFrame (with NaN) → (float32 array, bool missing mask)."""
    X = df.values.astype(np.float32)
    mask = np.isnan(X)   # True where missing
    return X, mask


def _initial_fill(X, mask, global_stats, columns):
    """Fill NaN positions with global mean (numeric) or mode (categorical)."""
    X_filled = X.copy()
    for j, col in enumerate(columns):
        missing_rows = np.where(mask[:, j])[0]
        if len(missing_rows) == 0:
            continue
        if global_stats[col]["categorical"]:
            fill = float(global_stats[col]["mode"])
        else:
            fill = (global_stats[col]["mean"] - global_stats[col]["min"]) / global_stats[col]["scale"]
        X_filled[missing_rows, j] = fill
    return X_filled


def _train_local(model, global_vec, X_filled, mask, lr, weight_decay, batch_size, local_epochs, fed_prox_mu):
    """Run local SGD for local_epochs; returns final avg loss."""
    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True
    )

    n = X_filled.shape[0]
    bs = min(batch_size, n)
    obs_mask = (~mask).astype(np.float32)  # 1 = observed, 0 = missing
    final_loss = 0.0

    for ep in trange(local_epochs, leave=False, colour='blue', desc='local epoch'):
        perm = np.random.permutation(n)
        n_batches = max(1, int(n / bs))
        X_shuf = X_filled[perm]
        M_shuf = obs_mask[perm]
        bX_list = np.array_split(X_shuf, n_batches)
        bM_list = np.array_split(M_shuf, n_batches)

        total_loss, total_iters = 0.0, 0
        for bX, bM in zip(bX_list, bM_list):
            optimizer.zero_grad()
            b_data = torch.from_numpy(bX).float().to(DEVICE)
            b_mask = torch.from_numpy(bM).float().to(DEVICE)
            loss, _ = model.compute_loss([b_data, b_mask])
            if fed_prox_mu > 0:
                local_vec = parameters_to_vector(model.parameters())
                # ||w - w^t||^2 (squared L2 norm, i.e. sum not mean) — matches
                # the FedProx objective exactly: F_k(w) + (mu/2)||w - w^t||^2
                # (Li et al., 2020). Disabled by default (fed_prox_mu=0) to
                # match the CAFE reference implementation — see _DEFAULTS.
                proximal_term = torch.square(local_vec - global_vec).sum()
                loss += (fed_prox_mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_iters += 1

        final_loss = total_loss / total_iters

    model.to("cpu")
    return final_loss


_IMPUTE_CHUNK = 256   # rows per chunk during imputation (L=1000 creates large tensors)


def _run_imputation(model, X_filled, mask):
    """Impute missing values with the current model; return filled array."""
    model.to(DEVICE)
    model.eval()

    obs_mask = (~mask).astype(np.float32)
    n = X_filled.shape[0]
    out = X_filled.copy()

    with torch.no_grad():
        for start in range(0, n, _IMPUTE_CHUNK):
            end = min(start + _IMPUTE_CHUNK, n)
            x_chunk = torch.from_numpy(X_filled[start:end]).float().to(DEVICE)
            m_chunk = torch.from_numpy(obs_mask[start:end]).float().to(DEVICE)
            out[start:end] = model.impute(x_chunk, m_chunk).detach().cpu().numpy()

    model.to("cpu")
    return out


def impute_miwae_file(dfs_dict_norm, args, global_stats=None):
    """
    Federated MIWAE imputation.

    Parameters
    ----------
    dfs_dict_norm  : dict {filename: DataFrame}
        One normalised DataFrame per client (NaN marks missing values).
    args           : argparse.Namespace (or any object with attributes)
        Hyperparameters — see _DEFAULTS above for supported keys.
    global_stats   : dict {col: {categorical, mean, median, iqr, mode}}
        Output of compute_global_stats() in impute_missing.py.
        Used for initial fill; if None, missing positions are zero-filled.

    Returns
    -------
    dict {filename: DataFrame}  — same structure, NaN positions filled.
    """
    # --- read hyperparams ---
    from data_utils import log_hps
    log_hps(args, _DEFAULTS, "Fed-MIWAE")
    global_rounds = _get(args, 'global_rounds')
    local_epochs = _get(args, 'local_epochs')
    lr = _get(args, 'lr')
    weight_decay = _get(args, 'weight_decay')
    batch_size = _get(args, 'batch_size')
    K = _get(args, 'K')
    L = _get(args, 'L')
    n_hidden = _get(args, 'n_hidden')
    latent_size = _get(args, 'latent_size')
    fed_prox_mu = _get(args, 'fed_prox_mu')
    seed = _get(args, 'seed')

    filenames = list(dfs_dict_norm.keys())
    columns = list(list(dfs_dict_norm.values())[0].columns)
    n_features = len(columns)

    # --- convert DataFrames to numpy and do initial fill ---
    client_X = {}  # training data (updated each round with latest imputations)
    client_mask = {}  # missing mask (fixed — observed positions never change)

    for fname, df in dfs_dict_norm.items():
        X, mask = _df_to_arrays(df)
        if global_stats is not None:
            X_filled = _initial_fill(X, mask, global_stats, columns)
        else:
            X_filled = np.nan_to_num(X, nan=0.0)
        client_X[fname] = X_filled
        client_mask[fname] = mask

    # --- initialise one MIWAE model per client (same architecture, different weights) ---
    torch.manual_seed(seed)
    models = {}
    for fname in filenames:
        m = MIWAE(num_features=n_features, K=K, L=L,
                  n_hidden=n_hidden, latent_size=latent_size, seed=seed)
        m.init(seed)
        models[fname] = m
    # --- initialise one MIWAE model for global model ---
    global_model = MIWAE(num_features=n_features, K=K, L=L,
                         n_hidden=n_hidden, latent_size=latent_size, seed=seed)
    global_model.init(seed)

    # --- federated loop ---
    for rnd in trange(1, global_rounds + 1, desc='Fed-MIWAE global round', colour='green'):
        tqdm.write(f"  [MIWAE] Round {rnd}/{global_rounds}")
        if fed_prox_mu > 0:
            global_model.to(DEVICE)
            global_model.eval()
            global_vec = parameters_to_vector(global_model.parameters()).detach()
        else:
            global_vec = None
        # 1. local training — each client trains on its locally-imputed data
        results = []
        for fname in filenames:
            _train_local(
                models[fname],
                global_vec,
                client_X[fname], client_mask[fname],
                lr, weight_decay, batch_size, local_epochs,
                fed_prox_mu
            )
            results.append({
                "params": models[fname].state_dict(),
                "num_steps": client_X[fname].shape[0],
                "client_id": fname,
                "round": rnd,
            })

        if fed_prox_mu > 0:
            global_model.to("cpu")

        # 2. FedAvg — aggregate weights
        global_weights = fedavg(results)
        global_model.load_state_dict(global_weights)
        # 3. broadcast global weights → re-impute → update training data
        for fname in filenames:
            models[fname].load_state_dict(global_weights)
            x_imp = _run_imputation(models[fname], client_X[fname], client_mask[fname])
            # only overwrite missing positions; observed values stay intact
            X_new = client_X[fname].copy()
            X_new[client_mask[fname]] = x_imp[client_mask[fname]]
            client_X[fname] = X_new

    # --- build output DataFrames ---
    dfs_dict_imputed = {}
    for fname, df in dfs_dict_norm.items():
        df_out = df.copy()
        x_imp = client_X[fname]
        for j, col in enumerate(columns):
            missing_rows = np.where(client_mask[fname][:, j])[0]
            if len(missing_rows) > 0:
                df_out.iloc[missing_rows, j] = x_imp[missing_rows, j]
        dfs_dict_imputed[fname] = df_out

    return dfs_dict_imputed
