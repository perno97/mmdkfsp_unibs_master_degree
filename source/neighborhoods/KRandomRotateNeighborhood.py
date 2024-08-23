import random
import time
from logging import Logger

from mkfsp.instance import Instance
from source.measures.Measures import Measures

Solution = list[int]


class KRandomRotateNeighborhood:
    """
    Class that implements the neighborhood use in the improvement phase of the VNDS algorithm.
    """
    def __init__(self, items_selection_counter_limit: int, unfeasible_move_items_limit: int,
                 use_weights_when_selecting_items: bool, use_weights_when_selecting_knapsacks: bool,
                 logger: Logger):
        self.use_weights_when_selecting_items: bool = use_weights_when_selecting_items
        self.use_weights_when_selecting_knapsacks: bool = use_weights_when_selecting_knapsacks
        self.items_selection_counter_limit = items_selection_counter_limit
        self.unfeasible_move_items_limit = unfeasible_move_items_limit
        self.logger = logger

    def get_best_neighbor(self,
                            instance: Instance,
                            input_solution: Solution,
                            input_solution_value: int,
                            k: int,
                            measures: Measures,
                            variable_indexes: list[int]
                            ) -> tuple[Solution, int]:
        """
        :param input_solution: The current solution
        :type input_solution: Solution
        :param variable_indexes: The indexes of the variable part of the solution
        :type variable_indexes: list[int]
        :param input_solution_value: The value of the current solution
        :type input_solution_value: int
        :param k: The number of items to move to generate a neighbor
        :type k: int
        :param measures: The measures of the execution
        :type measures: Measures
        :return: The best neighbor's variable part of the solution and its objective value
        :rtype: Solution
        """
        assert k <= len(variable_indexes), f"k={k} must be less than the length of the variable part of the solution"
        
        measures.get_best_neighbor_counter += 1

        # Save parameters
        measures.items_selection_counter_limit = self.items_selection_counter_limit
        measures.unfeasible_move_items_limit = self.unfeasible_move_items_limit
        measures.use_weights_when_selecting_items = self.use_weights_when_selecting_items

        # Initialize variables
        knapsacks_count = instance.n_knapsacks
        best_solution = input_solution.copy()
        best_value = input_solution_value

        # Iterate until a feasible solution is found or the limit of unfeasible items selections is reached
        # If best == True then stop only if the feasible solution found is better than the input solution
        is_feasible = False
        stop = False
        unfeasible_items_selections = 0
        while not stop and not self.check_unfeasible_items_selection_limit(unfeasible_items_selections, measures):
            if self.use_weights_when_selecting_items:
                # Give more weight to items with fewer moves
                weights: list[int] = []
                for item_index in variable_indexes:
                    if item_index not in measures.items_to_move_selected:
                        measures.items_to_move_selected[item_index] = 0
                    # Count how many times the item hasn't been selected
                    w = measures.item_selection_counter - measures.items_to_move_selected[item_index]
                    weights.append(1 + w * 10)
                indexes_to_change = random.choices(variable_indexes, weights=weights, k=k)
            else:
                # Pick a random selection of k items
                indexes_to_change = random.choices(variable_indexes, k=k)

            # Measure how many times an item has been selected
            for i in indexes_to_change:
                measures.item_selection_counter += 1
                measures.increment_item_to_move_selected(i)

            new_solution = input_solution.copy()

            # First, try the families' knapsacks of the items            
            item_family_dict = {}
            current_families_knapsacks = {}
            for i in indexes_to_change: # Cycle items to move
                family_index = len(instance.first_items) - 1
                item_family_dict[i] = family_index
                for index, first_item in enumerate(instance.first_items): # Get the family of the item
                    if first_item > i:
                        family_index = index - 1
                        item_family_dict[i] = family_index
                        break
                current_families_knapsacks[family_index] = measures.families_knapsacks[family_index]

            # Cycle until a feasible solution is found or there are no more possible knapsacks selections
            is_feasible = False
            unfeasible_move_items_counter = 0

            while (not is_feasible and
                   not self.check_unfeasible_move_items_limit(unfeasible_move_items_counter, measures)):
                self.logger.debug(f"Trying to move {len(indexes_to_change)} items to a different knapsack")

                # Try current families knapsacks for the first "unfeasible_move_items_limit - 1" times
                # No need to update the current families' knapsacks because any new knapsack is added
                # Update each item's knapsack
                for i in indexes_to_change:
                    family_index = item_family_dict[i]
                    if len(current_families_knapsacks[family_index]) == 1:
                        # If the family has only one knapsack then don't try to move it to the same knapsack again
                        try_current_families_knapsacks = False
                        break
                    #if try_current_families_knapsacks:
                    if self.use_weights_when_selecting_knapsacks:
                        weights: list[int] = []
                        for knapsack_index in current_families_knapsacks[family_index]:
                            if i not in measures.knapsack_selection_counter:
                                measures.knapsack_selection_counter[i] = {}
                            if knapsack_index not in measures.knapsack_selection_counter[i]:
                                measures.knapsack_selection_counter[i][knapsack_index] = 0
                            w = (measures.select_knapsack_counter -
                                    measures.knapsack_selection_counter[i][knapsack_index])
                            weights.append(1 + w * 10)
                        selected_knapsack = random.choices(
                            current_families_knapsacks[family_index], weights=weights
                        )[0]
                    else:
                        selected_knapsack = random.choices(current_families_knapsacks[family_index])[0]

                    measures.select_knapsack_counter += 1

                    new_solution[i] = selected_knapsack
                    measures.increment_knapsack_selection_counter(i, selected_knapsack)

                is_feasible, errors = instance.is_feasible(new_solution)
                if is_feasible:
                    new_value = instance.evaluate_solution(new_solution)
                    if new_value > best_value:  # This is a maximization problem
                        # This solution is better, then save it
                        best_value = new_value
                        best_solution = new_solution
                        stop = True
                        self.__update_families_knapsacks(instance, measures, new_solution)
                    else:
                        measures.discarded_get_best_neighbor += 1

                unfeasible_move_items_counter += 1
            unfeasible_items_selections += 1

        if not is_feasible:
            best_solution = input_solution
            best_value = input_solution_value
            self.logger.debug(f"No feasible neighbor found for the defined limits")
        return best_solution, best_value
    
    def __update_families_knapsacks(self, instance: Instance, measures: Measures, solution: Solution):
        for family_index in range(0, len(instance.first_items)):
            if family_index + 1 < len(instance.first_items):
                if family_index not in measures.families_knapsacks:
                    new_knapsacks = set(solution[instance.first_items[family_index]:instance.first_items[family_index + 1]])
                    new_knapsacks.discard(-1)
                    measures.families_knapsacks[family_index] = list(new_knapsacks)
                else:
                    current_knapsacks = measures.families_knapsacks[family_index]
                    new_knapsacks = set(solution[instance.first_items[family_index]:instance.first_items[family_index + 1]])
                    new_knapsacks.update(current_knapsacks)
                    new_knapsacks.discard(-1)
                    measures.families_knapsacks[family_index] = list(new_knapsacks)
            else:
                if family_index not in measures.families_knapsacks:
                    new_knapsacks = set(solution[instance.first_items[family_index]:])
                    new_knapsacks.discard(-1)
                    measures.families_knapsacks[family_index] = list(new_knapsacks)
                else:
                    current_knapsacks = measures.families_knapsacks[family_index]
                    new_knapsacks = set(solution[instance.first_items[family_index]:])
                    new_knapsacks.update(current_knapsacks)
                    new_knapsacks.discard(-1)
                    measures.families_knapsacks[family_index] = list(new_knapsacks)

    def check_unfeasible_items_selection_limit(self, unfeasible_items_selections: int, measures: Measures):
        limit_reached = False
        if (self.items_selection_counter_limit != 0 and
                unfeasible_items_selections >= self.items_selection_counter_limit):
            limit_reached = True
            measures.unfeasible_items_selection_limit_reached += 1
            self.logger.debug(f"Unfeasible items selection limit reached")

        return limit_reached

    def check_unfeasible_move_items_limit(self, unfeasible_move_items_counter: int, measures: Measures):
        limit_reached = False
        if (self.unfeasible_move_items_limit != 0 and
                unfeasible_move_items_counter >= self.unfeasible_move_items_limit):
            limit_reached = True
            measures.unfeasible_move_items_limit_reached += 1
            self.logger.debug(f"Unfeasible move items limit reached")

        return limit_reached
