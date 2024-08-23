from dataclasses import dataclass


@dataclass
class RelaxedSolutionInfo:
    xvars_dict: dict[int, float]
    xvars_rc_module_dict : dict[int, float]
    families_selection_score: dict[int, float]
    families_knapsack_selection_score: dict[int, list[float]]
    items_selection_score: dict[int, float]
    items_knapsack_selection_score: dict[int, list[float]]
    best_obj_value: float