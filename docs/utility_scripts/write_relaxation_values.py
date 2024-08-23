import logging
import os
from pathlib import Path

from mkfsp.instance import load_instance
from source.solvers.GurobiSolver import GurobiSolver


_project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Parameters
INSTANCES_DIR = Path(_project_dir, "instances")  # Path to the instances directory

# Get all the paths of the instances
paths: list[Path] = []
for dirpath, _, filenames in os.walk(INSTANCES_DIR):
    for fname in filenames:
        if fname.endswith('.json'):
            paths.append(Path(dirpath, fname))

path = paths[0]
instance = load_instance(path)

logger = logging.getLogger(__name__)
solver = GurobiSolver(logger)
xvars_dict, xvars_rc_module_dict,_, _, _, items_knapsack_selection_score, _, _ = solver.solve_relaxed_and_get_info(instance, None, 600)

sorted_xvars_dict = dict(sorted(xvars_dict.items(), key=lambda item: item[1], reverse=True))
sorted_xvars_rc_module_dict = dict(sorted(xvars_rc_module_dict.items(), key=lambda item: item[1]))

with open("families_selection_values.txt", "w") as file:
    for i,v in sorted_xvars_dict.items():
        file.write(f"x[{i}] - {v}\n")
    for i,v in sorted_xvars_rc_module_dict.items():
        file.write(f"RC x[{i}] - {v}\n")

with open("items_knapsacks.txt", "w") as file:
    for i,v in items_knapsack_selection_score.items():
        formatted_values = [f"{val:.4f}" for val in v]
        file.write(f"Item {i} - {formatted_values}\n")