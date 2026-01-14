from itertools import product
from typing import Any

import numpy as np
from jaxtyping import Array, Float


def select_best_params(
    results: dict[str, Any], best_metric: str = "value", nb_best: int = 4
) -> tuple[Float[Array, "nb_best 2"], Float[Array, "nb_best"], Float[Array, "nb_best"]]:
    """
    Find the best nb_best combinations of (T_d, B_s) according to the chosen best_metric
    ('value' or 'NLL'). Returns:
      - combos: array of shape (nb_best, 2), each row is (T_d, B_s)
      - combos_best_value: the minimum 'value' (across B_d) for each returned combo
      - combos_best_nll: the minimum 'NLL' (across B_d) for each returned combo
    in sorted order based on best_metric.
    """
    T_d_arr = np.array(results["T_d_patches"])
    B_s_arr = np.array(results["B_s_patches"])

    T_d_unique = np.sort(np.unique(T_d_arr))
    B_s_unique = np.sort(np.unique(B_s_arr))

    combos = []
    combos_best_value = []
    combos_best_nll = []

    # Loop over all (T_d, B_s) combos
    for T_d, B_s in product(T_d_unique, B_s_unique):
        (indices,) = np.where((T_d_arr == T_d) & (B_s_arr == B_s))

        # For each index, we have multiple noise evaluations for 'value' and 'NLL'
        value = results["value"][indices]  # shape (#B_d_points, #noise_runs)
        nll = results["NLL"][indices]

        mean_value = value.mean(axis=1)  # average over noise runs, shape (#B_d_points,)
        mean_nll = nll.mean(axis=1)

        # The "best" for that combo is the lowest across the B_d dimension
        min_value_for_combo = np.min(mean_value)
        min_nll_for_combo = np.min(mean_nll)

        combos.append((T_d, B_s))
        combos_best_value.append(min_value_for_combo)
        combos_best_nll.append(min_nll_for_combo)

    combos_arr = np.array(combos)
    combos_best_value_arr = np.array(combos_best_value)
    combos_best_nll_arr = np.array(combos_best_nll)

    # Sort combos by the chosen best_metric
    if best_metric == "value":
        sorted_idx = np.argsort(combos_best_value_arr)
    elif best_metric == "NLL":
        sorted_idx = np.argsort(combos_best_nll_arr)
    else:
        raise ValueError("best_metric must be 'value' or 'NLL'.")

    chosen_idx = sorted_idx[: min(nb_best, len(sorted_idx))]

    return (
        combos_arr[chosen_idx],
        combos_best_value_arr[chosen_idx],
        combos_best_nll_arr[chosen_idx],
    )
