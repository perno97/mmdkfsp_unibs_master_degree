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
    def __init__(self, instance, items_selection_counter_limit: int, unfeasible_move_items_limit: int,
                 use_weights_when_selecting_items: bool, use_weights_when_selecting_knapsacks: bool,
                 logger: Logger):
        self.use_weights_when_selecting_items: bool = use_weights_when_selecting_items
        self.use_weights_when_selecting_knapsacks: bool = use_weights_when_selecting_knapsacks
        self.instance: Instance = instance
        self.items_selection_counter_limit = items_selection_counter_limit
        self.unfeasible_move_items_limit = unfeasible_move_items_limit
        self.logger = logger

    def get_best_neighbor(self,
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

        # Initialize variables
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
                    weights.append(1 + w)
                indexes_to_change = random.choices(variable_indexes, weights=weights, k=k)
            else:
                # Pick a random selection of k items
                indexes_to_change = random.choices(variable_indexes, k=k)

            # Measure how many times an item has been selected
            for i in indexes_to_change:
                measures.item_selection_counter += 1
                measures.increment_item_to_move_selected(i)

            new_solution = input_solution.copy()

            # Cycle until a feasible solution is found or there are no more possible knapsacks selections
            is_feasible = False
            unfeasible_move_items_counter = 0

            while (not is_feasible and
                   not self.check_unfeasible_move_items_limit(unfeasible_move_items_counter, len(indexes_to_change), measures)):
                self.logger.debug(f"Trying to move {len(indexes_to_change)} items to a different knapsack")

                for i in indexes_to_change:
                    if self.use_weights_when_selecting_knapsacks:
                        weights: list[int] = []
                        for knapsack_index in range(self.instance.n_knapsacks):
                            if i not in measures.knapsack_selection_counter:
                                measures.knapsack_selection_counter[i] = {}
                            if knapsack_index not in measures.knapsack_selection_counter[i]:
                                measures.knapsack_selection_counter[i][knapsack_index] = 0
                            w = (measures.select_knapsack_counter -
                                    measures.knapsack_selection_counter[i][knapsack_index])
                            weights.append(1 + w)
                        selected_knapsack = random.choices(
                            range(self.instance.n_knapsacks), weights=weights
                        )[0]
                    else:
                        selected_knapsack = random.choices(range(self.instance.n_knapsacks))[0]

                    measures.select_knapsack_counter += 1

                    new_solution[i] = selected_knapsack
                    measures.increment_knapsack_selection_counter(i, selected_knapsack)

                is_feasible, _ = self.instance.is_feasible(new_solution)
                if is_feasible:
                    new_value = self.instance.evaluate_solution(new_solution)
                    if new_value > best_value:  # This is a maximization problem
                        best_value = new_value
                        best_solution = new_solution
                        stop = True
                    else:
                        measures.discarded_get_best_neighbor += 1

                unfeasible_move_items_counter += 1

            unfeasible_items_selections += 1

        if not is_feasible:
            best_solution = input_solution
            best_value = input_solution_value
            self.logger.debug(f"No feasible neighbor found for the defined limits")

        return best_solution, best_value

    def check_unfeasible_items_selection_limit(self, unfeasible_items_selections: int, measures: Measures):
        limit_reached = False
        if (self.items_selection_counter_limit != 0 and
                unfeasible_items_selections >= self.items_selection_counter_limit):
            limit_reached = True
            measures.unfeasible_items_selection_limit_reached += 1
            self.logger.debug(f"Unfeasible items selection limit reached")

        return limit_reached

    def check_unfeasible_move_items_limit(self, unfeasible_move_items_counter: int, number_of_indexes_to_change: int, measures: Measures):
        limit_reached = False
        if (self.unfeasible_move_items_limit != 0 and
                unfeasible_move_items_counter >= self.unfeasible_move_items_limit * number_of_indexes_to_change):
            limit_reached = True
            measures.unfeasible_move_items_limit_reached += 1
            self.logger.debug(f"Unfeasible move items limit reached")

        return limit_reached
