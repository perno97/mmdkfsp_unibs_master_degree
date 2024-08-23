import os
from pathlib import Path

from mkfsp.instance import load_instance


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
gurobi_solution = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 1, 1, 4, 1, 4, 4, 1, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 4, 4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, -1, -1, -1, -1, -1, 3, 3, 3, 3, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 4, 4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1]
vnds_solution = [-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
3,
3,
3,
3,
3,
3,
4,
1,
1,
1,
4,
4,
4,
4,
4,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
4,
4,
4,
4,
4,
4,
4,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
2,
2,
2,
2,
2,
2,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
0,
0,
0,
0,
0,
0,
0,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
4,
4,
4,
4,
4,
4,
3,
3,
3,
3,
3,
3,
3,
-1,
-1,
-1,
-1,
-1,
-1,
2,
2,
2,
2,
2,
2,
2,
2,
2,
2,
2,
2,
2,
2,
2,
2,
2,
2,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
4,
4,
0,
0,
0,
4,
4,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
0,
0,
0,
0,
0,
0,
0,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
3,
3,
3,
1,
3,
3,
3,
1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
0,
0,
0,
0,
0,
0,
0,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
3,
3,
3,
3,
3,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
1,
1,
1,
1,
1,
1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1,
-1
]

gurobi_families_with_splits = {}
gurobi_families_with_profit = {}
gurobi_families_with_penalty = {}
gurobi_families_with_profit_penalty_ratio = {}
gurobi_families_with_total_capacity = {}

vnds_families_with_splits = {}
vnds_families_with_profit = {}
vnds_families_with_penalty = {}
vnds_families_with_profit_penalty_ratio = {}
vnds_families_with_total_capacity = {}

only_gurobi_families_with_splits = {}
only_gurobi_families_with_profit = {}
only_gurobi_families_with_penalty = {}
only_gurobi_families_with_profit_penalty_ratio = {}
only_gurobi_families_with_total_capacity = {}

only_vnds_families_with_splits = {}
only_vnds_families_with_profit = {}
only_vnds_families_with_penalty = {}
only_vnds_families_with_profit_penalty_ratio = {}
only_vnds_families_with_total_capacity = {}

not_selected_families_with_profit = {}
not_selected_families_with_penalty = {}
not_selected_families_with_profit_penalty_ratio = {}
not_selected_families_with_total_capacity = {}

for family_index, first_item in enumerate(instance.first_items):
    last_item = instance.first_items[family_index + 1] if family_index + 1 < len(instance.first_items) else len(instance.items)
    not_selected = True
    only_selected_by_gurobi = True
    only_selected_by_vnds = True
    if gurobi_solution[first_item] != -1:
        only_selected_by_vnds = False
        not_selected = False
        gurobi_splits = set()
        for i in range(first_item, last_item):
            gurobi_splits.add(gurobi_solution[i])
        gurobi_families_with_splits[family_index] = len(gurobi_splits)
        gurobi_families_with_profit[family_index] = instance.profits[family_index]
        gurobi_families_with_penalty[family_index] = instance.penalties[family_index]
        gurobi_families_with_profit_penalty_ratio[family_index] = instance.profits[family_index] / instance.penalties[family_index]
        gurobi_families_with_total_capacity[family_index] = [sum(val) for val in zip(*instance.items[first_item:last_item])]
    if vnds_solution[first_item] != -1:
        only_selected_by_gurobi = False
        not_selected = False
        vnds_splits = set()
        for i in range(first_item, last_item):
            vnds_splits.add(vnds_solution[i])
        vnds_families_with_splits[family_index] = len(vnds_splits)
        vnds_families_with_profit[family_index] = instance.profits[family_index]
        vnds_families_with_penalty[family_index] = instance.penalties[family_index]
        vnds_families_with_profit_penalty_ratio[family_index] = instance.profits[family_index] / instance.penalties[family_index]
        vnds_families_with_total_capacity[family_index] = [sum(val) for val in zip(*instance.items[first_item:last_item])]
    if not not_selected and only_selected_by_gurobi:
        only_gurobi_splits = set()
        for i in range(first_item, last_item):
            only_gurobi_splits.add(gurobi_solution[i])
        only_gurobi_families_with_splits[family_index] = len(only_gurobi_splits)
        only_gurobi_families_with_profit[family_index] = instance.profits[family_index]
        only_gurobi_families_with_penalty[family_index] = instance.penalties[family_index]
        only_gurobi_families_with_profit_penalty_ratio[family_index] = instance.profits[family_index] / instance.penalties[family_index]
        only_gurobi_families_with_total_capacity[family_index] = [sum(val) for val in zip(*instance.items[first_item:last_item])]
    if not not_selected and only_selected_by_vnds:
        only_vnds_splits = set()
        for i in range(first_item, last_item):
            only_vnds_splits.add(vnds_solution[i])
        only_vnds_families_with_splits[family_index] = len(only_vnds_splits)
        only_vnds_families_with_profit[family_index] = instance.profits[family_index]
        only_vnds_families_with_penalty[family_index] = instance.penalties[family_index]
        only_vnds_families_with_profit_penalty_ratio[family_index] = instance.profits[family_index] / instance.penalties[family_index]
        only_vnds_families_with_total_capacity[family_index] = [sum(val) for val in zip(*instance.items[first_item:last_item])]
    if not_selected:
        not_selected_families_with_profit[family_index] = instance.profits[family_index]
        not_selected_families_with_penalty[family_index] = instance.penalties[family_index]
        not_selected_families_with_profit_penalty_ratio[family_index] = instance.profits[family_index] / instance.penalties[family_index]
        not_selected_families_with_total_capacity[family_index] = [sum(val) for val in zip(*instance.items[first_item:last_item])]

with open("compare_results.txt", "w") as file:
    file.write("Gurobi families:")
    file.write("\n")
    file.write(str([f"{family_index}: {n_splits}" for family_index, n_splits in gurobi_families_with_splits.items()]))
    file.write("\n")
    file.write("\tprofits: " + str([profit for profit in gurobi_families_with_profit.values()]))
    file.write("\n")
    file.write("\tpenalties: " + str([penalty for penalty in gurobi_families_with_penalty.values()]))
    file.write("\n")
    file.write("\tprofit/penalty: " + str([profit_penalty_ratio for profit_penalty_ratio in gurobi_families_with_profit_penalty_ratio.values()]))
    file.write("\n")
    file.write("\tTotal capacity")
    file.write("\n")
    for f,c in gurobi_families_with_total_capacity.items():
        file.write("\t" + f"{f}: {c}")
        file.write("\n")
    file.write("VNDS families:")
    file.write("\n")
    file.write(str([f"{family_index}: {n_splits}" for family_index, n_splits in vnds_families_with_splits.items()]))
    file.write("\n")
    file.write("\tprofits: " + str([profit for profit in vnds_families_with_profit.values()]))
    file.write("\n")
    file.write("\tpenalties: " + str([penalty for penalty in vnds_families_with_penalty.values()]))
    file.write("\n")
    file.write("\tprofit/penalty: " + str([profit_penalty_ratio for profit_penalty_ratio in vnds_families_with_profit_penalty_ratio.values()]))
    file.write("\n")
    file.write("\tTotal capacity")
    file.write("\n")
    for f,c in vnds_families_with_total_capacity.items():
        file.write("\t" + f"{f}: {c}")
        file.write("\n")
    file.write("Both Gurobi and VNDS families:")
    file.write("\n")
    file.write(str([f"{family_index}: {n_splits}" for family_index, n_splits in gurobi_families_with_splits.items() if family_index in vnds_families_with_splits]))
    file.write("\n")
    file.write("Only Gurobi families:")
    file.write("\n")
    file.write(str([f"{family_index}: {n_splits}" for family_index, n_splits in only_gurobi_families_with_splits.items()]))
    file.write("\n")
    file.write("\tprofits: " + str([profit for profit in only_gurobi_families_with_profit.values()]))
    file.write("\n")
    file.write("\tpenalties: " + str([penalty for penalty in only_gurobi_families_with_penalty.values()]))
    file.write("\n")
    file.write("\tprofit/penalty: " + str([profit_penalty_ratio for profit_penalty_ratio in only_gurobi_families_with_profit_penalty_ratio.values()]))
    file.write("\n")
    file.write("\tTotal capacity")
    file.write("\n")
    for f,c in only_gurobi_families_with_total_capacity.items():
        file.write("\t" + f"{f}: {c}")
        file.write("\n")
    file.write("Only VNDS families:")
    file.write("\n")
    file.write(str([f"{family_index}: {n_splits}" for family_index, n_splits in only_vnds_families_with_splits.items()]))
    file.write("\n")
    file.write("\tprofits: " + str([profit for profit in only_vnds_families_with_profit.values()]))
    file.write("\n")
    file.write("\tpenalties: " + str([penalty for penalty in only_vnds_families_with_penalty.values()]))
    file.write("\n")
    file.write("\tprofit/penalty: " + str([profit_penalty_ratio for profit_penalty_ratio in only_vnds_families_with_profit_penalty_ratio.values()]))
    file.write("\n")
    for f,c in only_vnds_families_with_total_capacity.items():
        file.write("\t" + f"{f}: {c}")
        file.write("\n")
    file.write("Not selected families:")
    file.write("\n")
    file.write("\tprofits: " + str([profit for profit in not_selected_families_with_profit.values()]))
    file.write("\n")
    file.write("\tpenalties: " + str([penalty for penalty in not_selected_families_with_penalty.values()]))
    file.write("\n")
    file.write("\tprofit/penalty: " + str([profit_penalty_ratio for profit_penalty_ratio in not_selected_families_with_profit_penalty_ratio.values()]))
    file.write("\n")
    for f,c in not_selected_families_with_total_capacity.items():
        file.write("\t" + f"{f}: {c}")
        file.write("\n")
