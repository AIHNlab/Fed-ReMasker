import argparse
import random
import numpy as np
import torch
from tqdm import trange, tqdm
from Remasker.flare_remasker import Flare_Remasker
from aggregation import fedavg, fedopt_update
from data_utils import get_hp
from torch.nn.utils import parameters_to_vector


_DEFAULTS = dict(
    global_rounds=60,     # reduced for benchmark feasibility (~25 hours total)
    local_epochs=5,       # 60 rounds x 5 epochs = 300 passes over the data.
                          # The clients partition the rows, so one round costs
                          # local_epochs passes over the full dataset -- which is
                          # what makes the centralized reference (one pooled
                          # client, same loop) an equal-budget comparison.
    # Flare_Remasker architecture / training defaults — from ReMasker paper Table 7
    batch_size=64,                 # paper: 64
    accum_iter=1,
    weight_decay=0.05,
    min_lr=1e-6,
    lr=None,
    blr=1e-3,                      # paper: lr = 1e-3
    warmup_epochs=4,
    embed_dim=64,                  # paper: embedding width = 64
    depth=8,                       # paper: encoder Transformer blocks = 8
    decoder_depth=4,               # paper: decoder Transformer blocks = 4
    num_heads=4,                   # paper: number of heads = 4
    mlp_ratio=4.0,
    encode_func="linear",          # paper: linear encoding function
    mask_ratio=0.3,                # paper: masking ratio = 0.3
    norm_field_loss=False,
    lambda_cat=0.01,    # categorical lambda
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed=0,
    # --- ablation: federated strategy — mutually exclusive (Li et al. 2020 / Reddi et al. 2020) ---
    strategy="fedavg",           # "fedavg" (default, unchanged behaviour) | "fedprox" | "fedopt"
    prox_mu=0.01,                 # FedProx proximal coefficient — only used when strategy="fedprox"
    server_opt_variant="adam",   # "adam" (FedAdam) | "yogi" (FedYogi) | "adagrad" (FedAdagrad) — only used when strategy="fedopt"
    server_lr=1e-2,               # FedOpt server learning rate (eta)
    server_beta1=0.9,             # server momentum decay
    server_beta2=0.99,            # server second-moment decay
    server_tau=1e-3,              # adaptivity/damping term
)


def _set_seed(seed):
    """
    Seed every RNG Fed-ReMasker draws from: weight initialisation, the
    RandomSampler batch order, and the per-forward masking noise in
    model_mae.py. Called once per dataset group so a group's result does not
    depend on how many groups ran before it (or on --para worker assignment).

    torch.use_deterministic_algorithms is deliberately not set here: some
    attention kernels have no deterministic CUDA implementation and would
    raise. cudnn.deterministic covers what can be covered; results are exact
    on a fixed environment, and stable to within kernel-level nondeterminism
    across GPUs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get(args, key):
    return get_hp(args, key, _DEFAULTS)


def _build_remasker_args(args):
    """Merge user-supplied args with defaults"""
    merged = {k: _get(args, k) for k in _DEFAULTS}
    # fields that must come from args (set by impute_missing.py)
    merged["cat_features"] = getattr(args, "cat_features", [])
    merged["categorical_vocab_sizes"] = getattr(args, "categorical_vocab_sizes", [])
    merged["data_dim"] = getattr(args, "data_dim", None)
    return argparse.Namespace(**merged)


def impute_remasker_file(dfs_dict_norm, args, global_stats=None):
    """
    Federated ReMasker imputation. --hp strategy=fedavg|fedprox|fedopt selects
    the federated strategy (mutually exclusive, default "fedavg"): "fedprox"
    adds a proximal term to each client's local loss (--hp prox_mu=X),
    "fedopt" runs a server-side adaptive-optimizer step on top of the FedAvg
    aggregate (Reddi et al. 2020) — see _DEFAULTS below.

    Parameters
    ----------
    dfs_dict_norm  : dict {filename: DataFrame}
        One normalised DataFrame per client (NaN marks missing values).
    args           : argparse.Namespace (or any object with attributes)
        Hyperparameters — see _DEFAULTS above for supported keys.
    global_stats   : dict, optional — not used by ReMasker (kept for API parity)

    Returns
    -------
    dict {filename: DataFrame}  — same structure, NaN positions filled.
    """
    from data_utils import log_hps
    log_hps(args, _DEFAULTS, "Fed-ReMasker")
    _set_seed(_get(args, "seed"))
    global_rounds = _get(args, "global_rounds")

    filenames = list(dfs_dict_norm.keys())
    remasker_args = _build_remasker_args(args)

    # --- one Flare_Remasker per client (keeps local norm parameters) ----------
    remaskers = {fname: Flare_Remasker(remasker_args) for fname in filenames}
    client_data = {
        fname: torch.tensor(df.values, dtype=torch.float32)
        for fname, df in dfs_dict_norm.items()
    }
    for fname, one_remasker in remaskers.items():
        one_remasker.prepare_data(client_data[fname])
    # --- initialise one Flare_Remasker model for global model ---
    global_model = Flare_Remasker(remasker_args)

    strategy = _get(args, "strategy")
    server_state = None  # FedOpt m/v state, persists across rounds below

    if strategy in ("fedopt", "fedprox"):
        # Each Flare_Remasker() call above (one per client, plus global_model)
        # gets its own independent random init — fine for plain FedAvg, where
        # global_model's weights are never referenced until round 1's
        # aggregate overwrites them. But fedopt's pseudo-gradient and
        # fedprox's proximal term both reference global_vec/global_model
        # starting in round 1, so without this sync each client would be
        # measured against an unrelated random point in round 1, not the
        # shared starting state either algorithm assumes. Only runs for
        # these two strategies — the default FedAvg path is untouched, so no
        # existing results need to be rerun.
        init_state = global_model.model.state_dict()
        for fname in filenames:
            remaskers[fname].model.load_state_dict(init_state)

    # --- federated loop -------------------------------------------------------
    for rnd in trange(1, global_rounds + 1, desc="Fed-ReMasker global round", colour="green"):
        tqdm.write(f"  [ReMasker] Round {rnd}/{global_rounds}")
        if strategy == "fedprox":
            global_model.to(remasker_args.device)
            global_model.model.eval()
            global_vec = parameters_to_vector(global_model.model.parameters()).detach()
        else:
            global_vec = None
        # 1. local training
        results = []
        for fname in filenames:
            remaskers[fname].local_train(global_vec, rnd)
            results.append({
                "params": remaskers[fname].model.state_dict(),
                "num_steps": client_data[fname].shape[0],
                "client_id": fname,
                "round": rnd,
            })

        # if strategy == "fedprox":
        #     global_model.to("cpu")

        # 2. Aggregate — plain FedAvg, or a server-side adaptive optimizer step on top
        new_avg_state = fedavg(results)
        if strategy == "fedopt":
            prev_global_state = {k: v.detach().clone() for k, v in global_model.model.state_dict().items()}
            global_weights, server_state = fedopt_update(
                prev_global_state, new_avg_state, server_state,
                variant=_get(args, "server_opt_variant"),
                server_lr=_get(args, "server_lr"),
                beta1=_get(args, "server_beta1"),
                beta2=_get(args, "server_beta2"),
                tau=_get(args, "server_tau"),
            )
        else:
            global_weights = new_avg_state
        global_model.model.load_state_dict(global_weights)
        # 3. broadcast
        for fname in filenames:
            remaskers[fname].model.load_state_dict(global_weights)

    # --- final imputation -----------------------------------------------------
    dfs_imputed = {}
    for fname, df in dfs_dict_norm.items():
        imputed = remaskers[fname].transform(client_data[fname]).detach().cpu().numpy()
        df_out = df.copy()
        mask = np.isnan(df.values)
        for j in range(df.shape[1]):
            missing_rows = np.where(mask[:, j])[0]
            if len(missing_rows):
                df_out.iloc[missing_rows, j] = imputed[missing_rows, j]
        dfs_imputed[fname] = df_out

    return dfs_imputed
