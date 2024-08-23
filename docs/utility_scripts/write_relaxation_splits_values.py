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
svars_dict, svars_rc_module_dict, best_obj_value, measures = solver.solve_relaxed_and_get_splits_vars(instance, None, 600)

sorted_svars_dict = dict(sorted(svars_dict.items(), key=lambda item: item[1], reverse=True))
sorted_svars_rc_module_dict = dict(sorted(svars_rc_module_dict.items(), key=lambda item: item[1]))

with open("families_splits.txt", "w") as file:
    for i,v in sorted_svars_dict.items():
        file.write(f"Family {i} splits - {v}\n")
    for i,v in sorted_svars_rc_module_dict.items():
        file.write(f"Family not split {i} RC - {v}\n")