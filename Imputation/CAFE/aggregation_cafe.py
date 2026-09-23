"""
CAFE aggregation -- direct implementation of Algorithm 2 from the paper:

    "Cafe: Improved Federated Data Imputation by Leveraging Missing Data
    Heterogeneity", Min et al., IEEE TKDE 2025.

The paper's Algorithm 2 (Cafe Averaging) is implemented as `cafe_avg()`.
The earlier clustering variants (`fedmechclw`, `fedmechw`) are kept as
optional alternatives but are NOT the method described in the paper.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


# ---------------------------------------------------------------------------
# Paper's Algorithm 2  (primary function)
# ---------------------------------------------------------------------------

def cafe_avg(
    weights: Dict[str, np.ndarray],
    missing_infos: Dict[str, dict],
    ms_coefs: Dict[str, np.ndarray],
    params: dict,
) -> List[np.ndarray]:
    """
    Cafe Averaging — Algorithm 2 from the CAFE paper.

    For each client k the personalised model is:

        theta_tilde^k = gamma * theta^k + (1 - gamma) * sum_{l!=k} w^{kl} * theta^l

    where
        c^{kl}  = 0.5*(1 - cosine_similarity(xi^k, xi^l))   complementarity score
        s^l     = N_f^l / max_j N_f^j                         sample-size score
        a^{kl}  = alpha * c^{kl} + (1 - alpha) * s^l          combined score
        w^{kl}  = (a^{kl})^beta / sum_{j!=k} (a^{kj})^beta    normalised weight

    Parameters
    ----------
    weights       : {client_id: model_weight_array}
    missing_infos : {client_id: info_dict}  -- output of ICEImputer.missing_info
                    Must contain key 'sample_size' (N_f^k in the paper).
    ms_coefs      : {client_id: ms_coef_array}  -- logistic mechanism model params
    params : dict with keys
        alpha  -- weight of complementarity vs sample size (paper default 0.95)
        beta   -- normalisation exponent                   (paper default 4)
        gamma  -- self-model weight                        (paper default 0.05)

    Returns
    -------
    List of personalised weight arrays, one per client, in insertion order.
    """
    weights_arr = np.array(list(weights.values()))        # (n, d)
    ms_coefs_arr = np.array(list(ms_coefs.values()))      # (n, d+1)

    alpha = params.get("alpha", 0.95)
    beta = params.get("beta", 4)
    gamma = params.get("gamma", 0.05)

    # --- sample-size scores: s^l = N_f^l / max_j N_f^j ---
    sample_sizes = np.array(
        [v["sample_size"] for v in missing_infos.values()], dtype=float
    )
    M_f = max(sample_sizes.max(), 1.0)
    s = sample_sizes / M_f                                # shape (n,)

    # --- complementarity matrix: c^{kl} = ½(1 − cosine_sim) ---
    C = _cos_sim_matrix(ms_coefs_arr)                     # shape (n, n), in [0,1]

    n = len(weights_arr)
    final_params = []

    for k in range(n):
        other = [l for l in range(n) if l != k]

        # combined score a^{kl} = alpha*c^{kl} + (1-alpha)*s^l
        a = np.array([alpha * C[k, l] + (1.0 - alpha) * s[l] for l in other])

        # normalised weights w^{kl} = (a^{kl})^beta / sum_j (a^{kj})^beta
        a_pow = np.power(np.maximum(a, 1e-8), beta)
        w = a_pow / a_pow.sum()

        # personalised model: theta_tilde^k = gamma*theta^k + (1-gamma)*sum_{l!=k} w^{kl}*theta^l
        other_avg = np.average(weights_arr[other], axis=0, weights=w)
        final_params.append(gamma * weights_arr[k] + (1.0 - gamma) * other_avg)

    return final_params


# ---------------------------------------------------------------------------
# Alternative aggregation variants (not described in paper, kept for reference)
# ---------------------------------------------------------------------------

def fedmechclw(
    weights: Dict[str, np.ndarray],
    missing_infos: Dict[str, dict],
    ms_coefs: Dict[str, np.ndarray],
    params: dict,
) -> List[np.ndarray]:
    """
    Cluster-based CAFE variant (NOT Algorithm 2 from the paper).
    Kept as an optional alternative.
    """
    weights_arr = np.array(list(weights.values()))
    sample_sizes = np.array([v["sample_row_pct"] + 1e-4 for v in missing_infos.values()])
    missing_pct = np.array([(1 - v["missing_cell_pct"]) + 1e-4 for v in missing_infos.values()])
    ms_coefs_arr = np.array(list(ms_coefs.values()))

    thres1 = params.get("thres1", 0.3)
    alpha = params.get("alpha", 0.95)
    scale_factor = params.get("beta", 4)
    client_thres = max(1, int(len(weights_arr) * params.get("client_thres", 1.0)))

    mech_sim_dist = _cos_sim_matrix(ms_coefs_arr)
    groups, centroids = _clustering(ms_coefs_arr, threshold=thres1)

    group_avg_params = []
    for group in groups:
        idx = np.array(group)
        w1 = sample_sizes[idx] ** scale_factor
        w2 = missing_pct[idx] ** scale_factor
        w1 /= w1.sum()
        w2 /= w2.sum()
        w = (alpha * w1 + (1 - alpha) * w2) ** scale_factor
        group_avg_params.append(np.average(weights_arr[idx], axis=0, weights=w))

    final_params = []
    for client_idx in range(len(weights_arr)):
        top_k_idx = set(np.argsort(mech_sim_dist[client_idx])[-client_thres:])
        filtered_centroids, filtered_group_params = [], []
        for group, centroid, g_param in zip(groups, centroids, group_avg_params):
            if top_k_idx.intersection(group):
                filtered_centroids.append(centroid)
                filtered_group_params.append(g_param)
        if not filtered_group_params:
            final_params.append(weights_arr[client_idx])
            continue
        distances = _cos_sim_distance(ms_coefs_arr[client_idx], filtered_centroids)
        distances = (distances + 1e-5) ** scale_factor
        final_params.append(np.average(filtered_group_params, axis=0, weights=distances))

    return final_params


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cos_sim_matrix(ms_coefs: np.ndarray) -> np.ndarray:
    """
    Pairwise complementarity matrix C where
        C[k,l] = 0.5*(1 - cosine_similarity(ms_coefs[k], ms_coefs[l]))
    matching Equation 1 in the paper.  Shape (n, n), values in [0, 1].
    """
    df = pd.DataFrame(ms_coefs).T    # (n_features, n_clients)
    sim = df.corr(
        method=lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)
    ).values                          # (n_clients, n_clients)
    return (1.0 - sim) / 2.0         # maps [-1,1] similarity → [0,1] distance


def _cos_sim_distance(coef_client: np.ndarray, coefs_centroid: List[np.ndarray]) -> np.ndarray:
    s2 = pd.Series(coef_client)
    dists = []
    for centroid in coefs_centroid:
        s1 = pd.Series(centroid)
        sim = s1.corr(
            s2, method=lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)
        )
        dists.append((1.0 - sim) / 2.0)
    return np.array(dists)


def _clustering(ms_coefs: np.ndarray, threshold: float = 0.1):
    n = len(ms_coefs)
    if n == 1:
        return [[0]], [ms_coefs[0]]
    try:
        agg = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=threshold,
        )
        labels = agg.fit_predict(ms_coefs)
    except Exception:
        labels = np.zeros(n, dtype=int)
    n_clusters = len(set(labels))
    groups = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        groups[label].append(i)
    centroids = [np.mean(ms_coefs[g], axis=0) for g in groups]
    return groups, centroids
