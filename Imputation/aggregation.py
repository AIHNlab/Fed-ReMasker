from typing import Any, Dict, List, Optional, Tuple
import torch


class WeightedAggregationHelper:
    """
    Accumulates weighted model contributions and returns the normalised aggregate.

    Usage
    -----
    helper = WeightedAggregationHelper()
    for client_result in results:
        helper.add(
            data=client_result["params"],
            weight=client_result["num_steps"],
            contributor_name=client_result.get("client_id"),
            contribution_round=client_result.get("round"),
        )
    global_weights = helper.get_result()
    """

    def __init__(self):
        self._total_weight: float = 0.0
        self._aggregated: Optional[Dict[str, torch.Tensor]] = None

    def add(
        self,
        data: Dict[str, torch.Tensor],
        weight: float,
        contributor_name: Optional[str] = None,
        contribution_round: Optional[int] = None,
    ) -> None:
        """Add one client's model weights with the given aggregation weight."""
        if weight <= 0:
            raise ValueError(
                f"weight must be positive, got {weight}" + (f" (contributor: {contributor_name})" if contributor_name else "")
            )
        if self._aggregated is None:
            self._aggregated = {k: v.float().clone() * weight for k, v in data.items()}
        else:
            for k in self._aggregated:
                self._aggregated[k].add_(data[k].float() * weight)
        self._total_weight += weight

    def get_result(self) -> Dict[str, torch.Tensor]:
        """Return the normalised (weighted-average) state dict."""
        if self._aggregated is None:
            raise RuntimeError("No contributions added — call add() at least once.")
        return {k: v / self._total_weight for k, v in self._aggregated.items()}


def fedavg(results: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Weighted FedAvg over a list of client results.

    Parameters
    ----------
    results : list of dicts, each containing:
        'params'    : state_dict (dict[str, Tensor]) — client model weights
        'num_steps' : int — number of training samples (aggregation weight)
        'client_id' : str, optional — client identifier
        'round'     : int, optional — current FL round number

    Returns
    -------
    Aggregated state_dict (dict[str, Tensor]).
    """
    if not results:
        raise ValueError("results list is empty")

    helper = WeightedAggregationHelper()
    for r in results:
        helper.add(
            data=r["params"],
            weight=float(r["num_steps"]),
            contributor_name=r.get("client_id"),
            contribution_round=r.get("round"),
        )
    return helper.get_result()


def fedopt_update(
    prev_global_state: Dict[str, torch.Tensor],
    new_avg_state: Dict[str, torch.Tensor],
    server_state: Optional[Dict[str, Dict[str, torch.Tensor]]],
    variant: str = "adam",
    server_lr: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.99,
    tau: float = 1e-3,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
    """
    Server-side Adaptive Federated Optimization (Reddi et al., 2020).

    `new_avg_state` (typically fedavg() over this round's client results) is
    treated as a pseudo-gradient relative to `prev_global_state`; the server
    then takes one Adam-style step using that pseudo-gradient, with momentum
    (m) / second-moment (v) estimates carried across FL rounds via
    `server_state` the same way Adam carries them across SGD steps.

    Parameters
    ----------
    prev_global_state : the global model's state_dict before this round.
    new_avg_state      : this round's FedAvg-aggregated client state_dict.
    server_state        : {"m": {...}, "v": {...}} from the previous round,
                          or None to initialise both to zero (round 1).
    variant             : "adam" (FedAdam, recommended default) | "yogi"
                          (FedYogi — gentler, more stable second-moment
                          update under noisy/non-convex updates) | "adagrad"
                          (FedAdagrad — monotonic accumulation, simplest but
                          can under-adapt over many rounds).
    server_lr, beta1, beta2, tau : standard Adam-family hyperparameters.

    Returns
    -------
    (new_global_state, new_server_state) — pass new_server_state back in as
    `server_state` on the next round.

    Runs under torch.no_grad(): m/v persist across all FL rounds (unlike
    plain fedavg(), which is recomputed fresh each round and never chains
    state), and the inputs are live model parameters with requires_grad=True.
    Without this, every round's arithmetic would attach to the autograd graph
    and — because m/v carry forward — chain across all rounds, growing an
    ever-larger unused graph instead of being freed each round.
    """
    with torch.no_grad():
        if server_state is None:
            server_state = {
                "m": {k: torch.zeros_like(v) for k, v in prev_global_state.items()},
                "v": {k: torch.zeros_like(v) for k, v in prev_global_state.items()},
            }
        m, v = server_state["m"], server_state["v"]

        new_global_state = {}
        for k in prev_global_state:
            delta = new_avg_state[k] - prev_global_state[k]
            m[k] = beta1 * m[k] + (1 - beta1) * delta
            if variant == "yogi":
                v[k] = v[k] - (1 - beta2) * torch.sign(v[k] - delta ** 2) * delta ** 2
            elif variant == "adagrad":
                v[k] = v[k] + delta ** 2
            else:  # "adam" / FedAdam
                v[k] = beta2 * v[k] + (1 - beta2) * delta ** 2
            new_global_state[k] = prev_global_state[k] + server_lr * m[k] / (torch.sqrt(v[k]) + tau)

        return new_global_state, {"m": m, "v": v}
