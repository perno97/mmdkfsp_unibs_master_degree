import random
from typing import Sequence

from source.solvers.GurobiSolver import GurobiSolver


Solution = Sequence[int]

class InitialSolutionBuilder:
    """
    Utility class to group the different methods used to build the initial solution.
    """
    def __init__(self, instance, cpu_time_limit: float):
        self.instance = instance
        self.cpu_time_limit = cpu_time_limit

    def build_initial_solutions(self) -> list[tuple[int, ...]]:
        """
        :return: the initial solution
        :rtype: Solution
        """
        items = self.instance.items
        first_items = self.instance.first_items

        # Initialize the list of the families indexes
        families_indexes = range(self.instance.n_families)
        initial_solutions: list[tuple[int, ...]] = []

        # Sort the families by the ratio between family profits and the sum of items weights for each resource
        for resource_index in range(self.instance.n_resources):
            sorted_families_indexes = families_indexes.copy()
            try:
                sorted_families_indexes.sort(
                    key=lambda fam_index: self.instance.profits[fam_index] / sum(
                        items[i_sort][resource_index] for i_sort in
                        range(
                            first_items[fam_index],
                            first_items[fam_index + 1] if fam_index + 1 < len(first_items) else len(items)
                        )
                    )
                )

                # Build the solution
                solution = self.build_solution(sorted_families_indexes)
                initial_solutions.append(tuple(solution))
            except ZeroDivisionError:
                # If the sum of the items weights for a resource is 0, then the families are not sorted
                pass

        if len(initial_solutions) == 0:
            # If the families cannot be sorted, then build a solution with random order
            random.shuffle(families_indexes)
            solution = self.__build_solution(families_indexes, first_items, items)
            initial_solutions.append(tuple(solution))
        return initial_solutions

    def build_initial_solutions_with_relaxed(self, gurobi_solver: GurobiSolver) -> Solution | None:
        """
        :return: the initial solution
        :rtype: Solution
        """

        relaxed_info, measures = gurobi_solver.solve_relaxed_and_get_info(
                instance=self.instance,
                cpu_time_limit=self.cpu_time_limit,
                output_path=None
            )
        xvars_dict = relaxed_info.xvars_dict
        xvars_rc_module_dict = relaxed_info.xvars_rc_module_dict
        families_selection_score = relaxed_info.families_selection_score
        families_knapsack_selection_score = relaxed_info.families_knapsack_selection_score
        items_selection_score = relaxed_info.items_selection_score
        items_knapsack_selection_score = relaxed_info.items_knapsack_selection_score
        best_obj_value = relaxed_info.best_obj_value
        
        sorted_families_indexes = sorted(xvars_dict, key=xvars_dict.get, reverse=True) + \
            sorted(xvars_rc_module_dict, key=xvars_rc_module_dict.get)

        # Build the solution
        solution = self.build_solution(sorted_families_indexes)        
        
        return solution, families_selection_score, families_knapsack_selection_score, items_selection_score, items_knapsack_selection_score, best_obj_value, measures
    
    def build_solution(self, families_indexes: list[int]) -> Solution:
        first_items = self.instance.first_items
        items = self.instance.items
        """
        Builds an initial solution by iteratively trying to add the families to the knapsacks.
        If an item of a family cannot be added to any knapsack, the algorithm stops and the whole family is
        put outside the knapsacks, then the algorithm tries to add the next family and so on.
        """
        # Initialize the solution to be an empty set of items
        solution: dict[int, int] = dict()

        # Initialize the list of the selected families
        selected_families: list[int] = []

        # Initialize the knapsacks to be the same as the instance
        # The values of temp_knapsacks will be decreased when an item is added to a knapsack
        # temp_knapsacks1: list[list[int]] = [list(seq) for seq in self.instance.knapsacks]

        temp_knapsacks1 = dict(sorted(enumerate(self.instance.knapsacks), key=lambda x: sum(x[1]), reverse=True))

        sorted_indexes_to_add = families_indexes.copy()
        not_selected_families = []
        for family in range(self.instance.n_families):
            if family not in families_indexes:
                not_selected_families.append(family)
        for family_index in sorted_indexes_to_add:  # Iterate through the families to add
            # Set boundaries for indexing the current family
            family_start_index = first_items[family_index]
            family_end_index = first_items[family_index + 1] if family_index + 1 < len(first_items) else len(items)

            # Start from the first item of the family
            item_index = family_start_index

            # Fix the knapsacks for the current family, in case the family is not selected
            temp_knapsacks2 = temp_knapsacks1.copy()

            family_can_be_selected = True
            family_solution = []

            # Iterate through the items of the family
            while family_can_be_selected and item_index < family_end_index:
                item = items[item_index]
                knapsack_found = False

                for index, knapsack in temp_knapsacks2.items():
                    constraint_violated = False
                    i = 0
                    while not constraint_violated and i < len(item):  # Iterate through the resources
                        if knapsack[i] < item[i]:  # Check if the knapsack can contain the item
                            constraint_violated = True  # The knapsack cannot contain the item
                        i += 1
                    if not constraint_violated:  # The knapsack can contain the item
                        knapsack_found = True
                        family_solution.append(index)
                        temp_knapsacks2[index] = [x - y for x, y in zip(knapsack, item)]
                    if knapsack_found:
                        break
                if not knapsack_found:
                    family_can_be_selected = False  # This causes to stop the iteration of the current family
                item_index += 1
            if family_can_be_selected:
                selected_families.append(family_index)
                # Update the temporary knapsacks values for next iteration
                temp_knapsacks1 = temp_knapsacks2
                # Add the family solution to the solution dictionary that maps
                # each item index to the knapsack index where it is loaded
                for i in range(family_start_index, family_end_index):
                    solution[i] = family_solution[i - family_start_index]
            else:
                # Add the family to the solution as not selected
                for i in range(family_start_index, family_end_index):
                    solution[i] = -1

            family_index += 1
        
        for family in not_selected_families:
            family_start_index = first_items[family]
            family_end_index = first_items[family + 1] if family + 1 < len(first_items) else len(items)
            for i in range(family_start_index, family_end_index):
                solution[i] = -1

        return [value for key, value in sorted(solution.items())]