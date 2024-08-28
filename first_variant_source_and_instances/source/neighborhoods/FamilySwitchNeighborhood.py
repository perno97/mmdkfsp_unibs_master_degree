import random
from logging import Logger

from mkfsp.instance import Instance

from source.measures.Measures import Measures

Solution = list[int]


class FamilySwitchNeighborhood:
    """
    Class that implements the neighborhood used in the shaking phase of the VNDS algorithm.
    """
    def __init__(self, instance, unfeasible_addition_limit: int, unfeasible_addition_knapsack_counter_limit: int,
                 use_weights_when_removing_families: bool, use_weights_when_selecting_knapsacks,
                 use_weights_when_adding_families: bool, logger: Logger, use_remove_counter_weights_removing: bool,
                 use_selection_weights_removing: bool, use_selection_weights_adding: bool,
                 use_remove_counter_weights_adding: bool):
        self.use_remove_counter_weights_adding: bool = use_remove_counter_weights_adding
        self.use_selection_weights_adding: bool = use_selection_weights_adding
        self.use_selection_weights_removing: bool = use_selection_weights_removing
        self.use_remove_counter_weights_removing: bool = use_remove_counter_weights_removing
        self.use_weights_when_adding_families: bool = use_weights_when_adding_families
        self.use_weights_when_selecting_knapsacks: bool = use_weights_when_selecting_knapsacks
        self.use_weights_when_removing_families: bool = use_weights_when_removing_families
        self.logger = logger
        self.instance: Instance = instance
        self.unfeasible_addition_limit: int = unfeasible_addition_limit
        self.unfeasible_addition_knapsack_counter_limit: int = unfeasible_addition_knapsack_counter_limit

    def get_random_neighbor(
            self,
            input_solution: Solution,
            k: int,
            measures: Measures
    ) -> Solution | None:
        """
        Get a random neighbor of the input solution by removing a family of items and adding another one,
        which wasn't already in the solution.
        :param input_solution: Input solution.
        :param k: Number of families to move.
        :return: Random neighbor and the indexes of the items that have been moved.
        """
        # Define which families are in the solution and which are not.
        selected_families, not_selected_families = self.__get_selected_families(input_solution)

        for f in selected_families:
            measures.increment_selected_families_counter(f)

        for f in not_selected_families:
            measures.increment_not_selected_families_counter(f)

        self.logger.debug(f"Removing families")

        solution_with_removed_families = self.remove_families(
            input_solution=input_solution,
            selected_families=selected_families,
            k=k,
            measures=measures
        )

        self.logger.debug(f"Adding families")

        if len(not_selected_families) != 0:
            solution_with_added_families = self.add_families(
                input_solution=solution_with_removed_families,
                not_selected_families=not_selected_families,
                measures=measures
            )
            solution_to_return = solution_with_added_families
        else:
            if any(x != -1 for x in solution_with_removed_families):
                solution_to_return = solution_with_removed_families
            else:
                self.logger.debug(f"The solution is empty, returning the input solution")
                # Return the input solution because otherwise there won't be any item to move in the improving phase
                solution_to_return = input_solution
        return solution_to_return

    def __get_selected_families(self, input_solution) -> tuple[list[int], list[int]]:
        """
        Get the indexes of the families that are in the solution.
        :param input_solution: Input solution.
        :return: Indexes of the families that are in the solution.
        :rtype: list[int]
        """
        selected_families: list[int] = []
        not_selected_families: list[int] = []
        family_index = 0
        for first_item_index in self.instance.first_items:
            if input_solution[first_item_index] >= 0:
                selected_families.append(family_index)
            else:
                not_selected_families.append(family_index)
            family_index += 1
        return selected_families, not_selected_families

    def remove_families(self, input_solution: Solution,
                        selected_families: list[int],
                        k: int,
                        measures: Measures) -> Solution:
        solution_with_removed_families = input_solution.copy()
        # Select a random tuple for removal
        if len(selected_families) != 0:
            removal_number = min(k, len(selected_families))
            measures.increment_removal_number_values_counter(removal_number)
            if self.use_weights_when_removing_families:
                if self.use_selection_weights_removing:
                    weights: list[int] = []
                    for family_index in selected_families:
                        if family_index not in measures.selected_families_counter:
                            measures.selected_families_counter[family_index] = 0
                        w = measures.selected_families_counter[family_index]
                        weights.append(1 + w)
                    families_to_remove = random.choices(selected_families, weights=weights, k=removal_number)
                elif self.use_remove_counter_weights_removing:
                    weights: list[int] = []
                    for family_index in selected_families:
                        if family_index not in measures.removed_families_counter:
                            measures.removed_families_counter[family_index] = 0
                        w = (measures.remove_counter -
                             measures.removed_families_counter[family_index])
                        weights.append(1 + w)
                    families_to_remove = random.choices(
                        selected_families,
                        weights=weights,
                        k=removal_number
                    )
                else:
                    weights: list[int] = []
                    max_profit = max(self.instance.profits)
                    for family in selected_families:
                        weights.append(1 + max_profit - self.instance.profits[family])
                    families_to_remove = random.choices(selected_families, weights=weights, k=removal_number)
            else:
                families_to_remove = random.choices(selected_families, k=removal_number)

            # Remove the selected families.
            for family_index in families_to_remove:
                measures.increment_removed_families_counter(family_index)
                family_start_index = self.instance.first_items[family_index]
                family_end_index = self.instance.first_items[family_index + 1] \
                    if family_index + 1 < len(self.instance.first_items) else len(self.instance.items)
                for item_index in range(family_start_index, family_end_index):
                    solution_with_removed_families[item_index] = -1

            measures.remove_counter += 1
        else:
            measures.no_families_to_remove_counter += 1
            self.logger.debug("No families to remove")

        return solution_with_removed_families

    def add_families(self, input_solution: Solution, not_selected_families: list[int],
                        measures: Measures) -> tuple[Solution, bool]:
        solution_with_added_families = input_solution.copy()
        added_families = []

        # The number of possible additions can be very high, so we need to set a limit of tries
        unfeasible_addition_counter = 0

        is_feasible = True
        while is_feasible and not self.__check_unfeasible_addition_counter_limit(
                unfeasible_addition_counter, measures):
            # Select a random family for addition
            family_index = self.__select_random_family_for_addition(not_selected_families, measures)
            while (family_index in added_families and
                   not self.__check_unfeasible_addition_counter_limit(
                       unfeasible_addition_counter, measures
                   )):
                # Select another family if the selected one has already been added
                self.logger.debug(f"Family {family_index} has already been added, retrying...")
                # Increment the counter of unfeasible additions if the family has already been added
                unfeasible_addition_counter += 1
                family_index = self.__select_random_family_for_addition(not_selected_families, measures)

            solution_to_try, is_solution_to_try_feasible = self.__try_to_add_family(
                solution_with_added_families,
                family_index,
                measures
            )

            if is_solution_to_try_feasible:
                solution_with_added_families = solution_to_try.copy()
                measures.increment_family_addition_counter(family_index)
                added_families.append(family_index)
                measures.add_family_counter += 1
                is_feasible = is_solution_to_try_feasible
            else:
                unfeasible_addition_counter += 1

        self.logger.debug(f"Added {len(added_families)} families up to an unfeasible solution")

        # Try to add a random family until a feasible solution is found or the limit is reached
        while not is_feasible and not self.__check_unfeasible_addition_counter_limit(
                unfeasible_addition_counter, measures
        ):
            # Select a random family for addition
            family_index = self.__select_random_family_for_addition(not_selected_families, measures)

            solution_to_try, is_solution_to_try_feasible = self.__try_to_add_family(
                solution_with_added_families,
                family_index,
                measures
            )

            if is_solution_to_try_feasible:
                solution_with_added_families = solution_to_try.copy()
                measures.increment_family_addition_counter(family_index)
                is_feasible = is_solution_to_try_feasible

            # Update termination conditions for this loop
            if not is_feasible:
                unfeasible_addition_counter += 1
                measures.unfeasible_add_random_families += 1
                self.logger.debug("Generated an unfeasible neighbor, retrying...")

        return solution_with_added_families

    def __check_unfeasible_addition_knapsack_counter_limit(self, unfeasible_addition_knapsack_counter: int,
                                                           measures: Measures) -> bool:
        limit_reached = False
        if (self.unfeasible_addition_knapsack_counter_limit != 0 and
                unfeasible_addition_knapsack_counter >= self.unfeasible_addition_knapsack_counter_limit):
            limit_reached = True
            measures.unfeasible_addition_knapsack_counter_limit_reached += 1
        return limit_reached

    def __check_unfeasible_addition_counter_limit(self, unfeasible_addition_counter: int,
                                                  measures: Measures) -> bool:
        limit_reached = False
        if (self.unfeasible_addition_limit != 0 and
                unfeasible_addition_counter >= self.unfeasible_addition_limit):
            limit_reached = True
            measures.unfeasible_addition_counter_limit_reached += 1
            self.logger.debug("Addition counter limit reached")
        return limit_reached

    def __try_to_add_family(self, solution_with_added_families: Solution, family_index: int,
                                measures: Measures) -> tuple[Solution, bool]:
        self.logger.debug(f"Trying to add family {family_index}")
        # Initialize the solution to the one with removed families
        solution_to_try = solution_with_added_families.copy()

        # Measure how many times a family has been selected
        measures.increment_try_family_add_counter(family_index)

        # Try adding this family
        family_start_index = self.instance.first_items[family_index]
        family_end_index = self.instance.first_items[family_index + 1] \
            if family_index + 1 < len(self.instance.first_items) else len(self.instance.items)

        # Search for a feasible solution by moving the items of the added family
        indexes_to_change = range(family_start_index, family_end_index)

        # Cycle until a feasible solution is found or the limit of unfeasible knapsack selections is reached
        is_solution_to_try_feasible = False
        unfeasible_addition_knapsack_counter = 0
        family_knapsacks = []
        while (not is_solution_to_try_feasible and
               not self.__check_unfeasible_addition_knapsack_counter_limit(
                   unfeasible_addition_knapsack_counter, measures
               )):
            # Increment the counter of knapsack selections
            add_new_knapsack = True
            # Try among family knapsacks for "remaining_attempts" times, then add a new knapsack and retry
            # if len(family_knapsacks) == 0, it adds a new knapsack at the first iteration
            remaining_attempts = len(family_knapsacks) * len(indexes_to_change) if len(family_knapsacks) > 0 else 1
            # Move items of the family to the selected knapsacks
            for i in indexes_to_change:
                # Always executed at first iteration of "for" cycle
                # At the first "while" iteration, family_knapsacks is empty
                # It adds a new knapsack if the previous selection ("for" cycle) wasn't feasible
                if add_new_knapsack and remaining_attempts > 0:
                    # Select a random knapsack
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
                            range(self.instance.n_knapsacks),
                            weights=weights
                        )[0]
                    else:
                        selected_knapsack = random.choice(range(self.instance.n_knapsacks))

                    family_knapsacks.append(selected_knapsack)
                    add_new_knapsack = False
                else:
                    # Select a random knapsack among family_knapsacks
                    selected_knapsack = random.choice(family_knapsacks)

                measures.select_knapsack_counter += 1
                measures.increment_knapsack_selection_counter(i, selected_knapsack)

                solution_to_try[i] = selected_knapsack
            is_solution_to_try_feasible, _ = self.instance.is_feasible(solution_to_try)
            if is_solution_to_try_feasible:
                measures.families_knapsacks[family_index] = family_knapsacks
            else:
                unfeasible_addition_knapsack_counter += 1

        return solution_to_try, is_solution_to_try_feasible

    def __select_random_family_for_addition(
            self, not_selected_families: list[int], measures: Measures) -> int:
        if self.use_weights_when_adding_families:
            if self.use_selection_weights_adding:
                weights: list[int] = []
                for family_index in not_selected_families:
                    if family_index not in measures.not_selected_families_counter:
                        measures.not_selected_families_counter[family_index] = 0
                    w = measures.not_selected_families_counter[family_index]
                    weights.append(1 + w)
                selected_family = random.choices(not_selected_families, weights=weights)[0]
            elif self.use_remove_counter_weights_adding:
                weights: list[int] = []
                for family_index in not_selected_families:
                    if family_index not in measures.removed_families_counter:
                        measures.removed_families_counter[family_index] = 0
                    w = measures.removed_families_counter[family_index]
                    weights.append(1 + w)
                selected_family = random.choices(not_selected_families, weights=weights)[0]
            else:
                weights: list[int] = []
                for family in not_selected_families:
                    weights.append(1 + self.instance.profits[family])
                selected_family = random.choices(not_selected_families, weights=weights)[0]
        else:
            selected_family = random.choice(not_selected_families)
        return selected_family
