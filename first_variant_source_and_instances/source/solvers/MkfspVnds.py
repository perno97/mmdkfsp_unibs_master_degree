from logging import Logger
import random
import time
from typing import Sequence

from mkfsp.model import build_model
from source.data_structures.SolutionValueTime import SolutionValueTime
from source.measures.Measures import Measures
from source.neighborhoods.FamilySwitchNeighborhood import FamilySwitchNeighborhood
from source.neighborhoods.KRandomRotateNeighborhood import KRandomRotateNeighborhood
from mkfsp.instance import Instance
from source.solvers.GurobiSolver import GurobiSolver

Solution = list[int]  # Defines a solution as sequence of integers
INT_TOLERANCE = 1e-6

class MkfspVnds:
    """
    Class representing a solver of the mkfsp that uses the VNDS algorithm.
    """

    def __init__(
            self,
            logger: Logger,
            k_max: int,
            family_switch_neighborhood: FamilySwitchNeighborhood,
            k_random_rotate_neighborhood: KRandomRotateNeighborhood,
            instance: Instance,
            cpu_time_limit: float,
            iterations_counter_limit: int,
            iterations_without_improvement_limit: int,
            iterations_without_improvement_before_reset: int,
            mip_gap: float | None = None
            ):
        self.iterations_without_improvement_before_reset: int = iterations_without_improvement_before_reset
        self.iterations_without_improvement_from_last_reset_counter: int = 0
        self.mip_gap: float | None = mip_gap
        self.relaxed_optimum_value: int = 0
        self.logger: Logger = logger
        self.iterations_without_improvement_limit: int = iterations_without_improvement_limit
        self.iterations_without_improvement: int = 0
        self.k_max: int = k_max
        self.cpu_time_limit: float = cpu_time_limit
        self.iterations_counter_limit: int = iterations_counter_limit
        self.vns_neighborhood: FamilySwitchNeighborhood = family_switch_neighborhood
        self.vnd_neighborhood: KRandomRotateNeighborhood = k_random_rotate_neighborhood
        self.instance: Instance = instance

    def solve(self) \
            -> tuple[
                Solution,
                int,
                Measures,
            ]:
        """
        Solves the instance using the VNDS algorithm.
        :return: the best solution found and its objective value
        :rtype: tuple[Solution, int]
        """

        # Initialize the measures
        measures: Measures = Measures()

        # Save parameters
        measures.k_max = self.k_max
        measures.vnds_mip_gap = self.mip_gap
        measures.iterations_without_improvement_limit = self.iterations_without_improvement_limit
        measures.cpu_time_limit = self.cpu_time_limit
        measures.iterations_counter_limit = self.iterations_counter_limit
        measures.iterations_without_improvement_before_reset = self.iterations_without_improvement_before_reset
        measures.items_selection_counter_limit = self.vnd_neighborhood.items_selection_counter_limit
        measures.unfeasible_move_items_limit = self.vnd_neighborhood.unfeasible_move_items_limit
        measures.use_weights_when_selecting_items = self.vnd_neighborhood.use_weights_when_selecting_items
        measures.unfeasible_addition_limit = self.vns_neighborhood.unfeasible_addition_limit
        measures.unfeasible_addition_knapsack_counter_limit = self.vns_neighborhood.unfeasible_addition_knapsack_counter_limit
        measures.use_weights_when_adding_families = self.vns_neighborhood.use_weights_when_adding_families
        measures.use_weights_when_selecting_knapsacks = self.vns_neighborhood.use_weights_when_selecting_knapsacks
        measures.use_weights_when_removing_families = self.vns_neighborhood.use_weights_when_removing_families
        measures.use_remove_counter_weights_adding = self.vns_neighborhood.use_remove_counter_weights_adding
        measures.use_selection_weights_adding = self.vns_neighborhood.use_selection_weights_adding
        measures.use_selection_weights_removing = self.vns_neighborhood.use_selection_weights_removing
        measures.use_remove_counter_weights_removing = self.vns_neighborhood.use_remove_counter_weights_removing

        # Start measuring execution time
        self.logger.info("Starting VNDS algorithm")
        measures.start_time = time.perf_counter()

        # Compute the relaxed optimum value
        if self.mip_gap is not None:
            measures.relaxed_compute_time = time.perf_counter()
            # Solve the relaxed problem with Gurobi to get an upper bound
            gurobi_solver = GurobiSolver(
                logger=self.logger
            )
            solutions, relaxed_measures = gurobi_solver.solve(
                instance=self.instance,
                cpu_time_limit=self.cpu_time_limit,
                output_path=None,
                relaxed=True
            )
            measures.relaxed_compute_time = time.perf_counter() - measures.relaxed_compute_time
            if relaxed_measures.stopping_cause != Measures.STOP_GUROBI_OPTIMAL:
                self.logger.warning("The relaxed problem is not optimal")
                self.mip_gap = None
            else:
                self.logger.info(f"Relaxed solution value: {str(solutions[0][1])}"
                                 f"\t\t\t\t\t\t\t{relaxed_measures.execution_time:.4f}s")
                self.relaxed_optimum_value = max(solutions, key=lambda x: x[1])[1]
                measures.relaxed_optimum_value = self.relaxed_optimum_value

        # Initialize the number of iterations without improvement
        self.iterations_without_improvement = 0

        # Build initial solution
        self.logger.debug("Building initial solution")
        measures.building_solution_time = time.perf_counter()

        measures.best_solution = self.build_initial_solutions_with_relaxed()
        measures.building_solution_time = time.perf_counter() - measures.building_solution_time
        measures.best_solution_value = self.instance.evaluate_solution(measures.best_solution)
        measures.initial_solution_obj_value = measures.best_solution_value

        self.logger.info(f"Initial solution value: {str(measures.best_solution_value)}"
                         f"\t\t\t\t\t\t\t{measures.building_solution_time:.4f}s")

        # Memorize current solution families' knapsacks
        self.__update_families_knapsacks(measures, measures.best_solution)

        measures.top_solutions.append(
            SolutionValueTime(measures.best_solution, measures.best_solution_value, time.perf_counter() - measures.start_time)
        )

        # Initialize counter
        measures.reached_kmax_counter = 0

        # Main loop
        while not self.check_stopping_criterion(measures):
            self.logger.debug(f"Reached k_max {str(measures.reached_kmax_counter)} times")
            k = 1
            measures.reached_kmax_counter += 1
            while k <= self.k_max:
                measures.iterations_counter += 1
                self.logger.debug(f"Iteration: {str(measures.iterations_counter)}")

                measures.increment_vnds_k_values_counter(k)

                self.logger.debug(f"VNDS k value: {str(k)}")

                if 0 < self.iterations_without_improvement_before_reset <= self.iterations_without_improvement_from_last_reset_counter:
                    self.iterations_without_improvement_from_last_reset_counter = 0
                    measures.improve_top_solution_counter += 1
                    x_shake = measures.best_solution
                    measures.best_solution_value
                else:
                    x_shake, x_shake_value = self.normal_shake(measures, k)

                # Pick only the indexes of items that are selected                
                valid_indexes = [i for i, x in enumerate(x_shake) if x != -1]

                # Memorize which families have splits
                # Compute knapsacks available capacity considering only families with no splits
                split_families = {}
                fixed_families = list(range(0, len(self.instance.first_items)))
                temp_knapsacks = list(self.instance.knapsacks).copy()
                for family_index, first_item in enumerate(self.instance.first_items):  # Cycle families
                    if x_shake[first_item] != -1:  # The family is selected
                        last_item = self.instance.first_items[family_index + 1] - 1 if family_index + 1 < len(
                            self.instance.first_items) else len(x_shake) - 1
                        if len(measures.families_knapsacks[family_index]) > 1:  # The family has a split
                            split_families[family_index] = [first_item, last_item]
                            fixed_families.remove(family_index)
                        else:
                            for item_index, item_values in enumerate(self.instance.items[first_item:last_item]):
                                knapsack = temp_knapsacks[x_shake[item_index + first_item]]
                                temp_knapsacks[x_shake[item_index + first_item]] = [x - y for x, y in
                                                                                    zip(knapsack, item_values)]
                    else:
                        # Fix the family out of the solution
                        fixed_families.remove(family_index)

                possible_knapsacks_per_family_dict = {}
                for family_index, first_and_last_items in split_families.items():  # Cycle families
                    first_item = first_and_last_items[0]
                    if x_shake[first_item] != -1:  # The family is selected
                        if len(measures.families_knapsacks[family_index]) > 1:  # The family has a split
                            last_item = first_and_last_items[1]
                            family_capacity = [sum(valori) for valori in
                                               zip(*self.instance.items[first_item:last_item])]
                            possible_knapsacks = {knapsack_index: knapsack for knapsack_index, knapsack in
                                                  enumerate(temp_knapsacks) if all(
                                    knapsack[i] > family_capacity[i] for i in range(len(family_capacity)))}
                            if len(possible_knapsacks) > 0:
                                possible_knapsacks_per_family_dict[family_index] = possible_knapsacks

                # Initialize solution to improve
                # Evaluate x_shake_value because if the improvement returns an unfeasible solution then
                # it will check the x_shake solution and its value in the neighborhood change step
                x_improvement = x_shake.copy()
                x_shake_value = self.instance.evaluate_solution(x_shake)
                x_improvement_value = x_shake_value
                improvement_time = 0.0

                if len(possible_knapsacks_per_family_dict) != 0:
                    # Pick a random family
                    keys = list(possible_knapsacks_per_family_dict.keys())
                    selected_family = random.choice(keys)
                    knapsack_indexes = list(possible_knapsacks_per_family_dict[selected_family].keys())
                    selected_knapsack = random.choice(knapsack_indexes)

                    # Fix all the items of the family to the selected knapsack
                    first_and_last_items = split_families[selected_family]
                    for i in range(first_and_last_items[0], first_and_last_items[1]):
                        x_improvement[i] = selected_knapsack
                        valid_indexes.remove(i)

                    # If the solution is empty, skip the improvement phase
                    if len(valid_indexes) > 0:
                        variable_indexes = random.sample(valid_indexes, min(k, len(valid_indexes)))

                        self.logger.debug(f"Variable indexes: {str(variable_indexes)}")

                        # Improve
                        # Implicit decomposition by passing variable_indexes
                        improvement_time = time.perf_counter()
                        x_improvement, x_improvement_value = self.improve(
                            x_improvement, x_improvement_value, self.vnd_neighborhood, variable_indexes, measures
                        )   

                        improvement_time = time.perf_counter() - improvement_time

                is_feasible_check, _ = self.instance.is_feasible(x_improvement)
                if not is_feasible_check:
                    x_improvement = x_shake
                    x_improvement_value = x_shake_value

                measures.best_solution, measures.best_solution_value, k, has_improved = (
                    self.neighborhood_change_step(
                        measures.best_solution,
                        x_improvement,
                        measures.best_solution_value,
                        x_improvement_value,
                        k
                    )
                )
                if has_improved:
                    self.iterations_without_improvement = 0
                    self.iterations_without_improvement_from_last_reset_counter = 0
                    self.logger.info(f"New best solution found: {str(measures.best_solution_value)}"
                                     f"\t\t\t\t\t\t\t{time.perf_counter() - measures.start_time:.4f}s")
                    measures.cumulative_improvement_found_time += improvement_time
                    measures.improvements_counter += 1
                    measures.top_solutions.append(
                        SolutionValueTime(measures.best_solution, measures.best_solution_value, time.perf_counter() - measures.start_time)
                    )
                else:
                    self.iterations_without_improvement += 1
                    self.iterations_without_improvement_from_last_reset_counter += 1
                    measures.cumulative_improvement_not_found_time += improvement_time
                    measures.not_improvements_counter += 1
                self.logger.debug(f"Current solution value: {str(measures.best_solution_value)}")
                self.logger.debug(f"VNDS new k value: {str(k)}")
        return (
            measures.best_solution,
            measures.best_solution_value,
            measures
        )

    def __update_families_knapsacks(self, measures: Measures, solution: Solution):
        for family_index in range(0, len(self.instance.first_items)):
            if family_index + 1 < len(self.instance.first_items):
                if family_index not in measures.families_knapsacks:
                    new_knapsacks = set(
                        solution[self.instance.first_items[family_index]:self.instance.first_items[family_index + 1]])
                    new_knapsacks.discard(-1)
                    measures.families_knapsacks[family_index] = list(new_knapsacks)
                else:
                    current_knapsacks = measures.families_knapsacks[family_index]
                    new_knapsacks = set(
                        solution[self.instance.first_items[family_index]:self.instance.first_items[family_index + 1]])
                    new_knapsacks.update(current_knapsacks)
                    new_knapsacks.discard(-1)
                    measures.families_knapsacks[family_index] = list(new_knapsacks)
            else:
                if family_index not in measures.families_knapsacks:
                    new_knapsacks = set(solution[self.instance.first_items[family_index]:])
                    new_knapsacks.discard(-1)
                    measures.families_knapsacks[family_index] = list(new_knapsacks)
                else:
                    current_knapsacks = measures.families_knapsacks[family_index]
                    new_knapsacks = set(solution[self.instance.first_items[family_index]:])
                    new_knapsacks.update(current_knapsacks)
                    new_knapsacks.discard(-1)
                    measures.families_knapsacks[family_index] = list(new_knapsacks)

    def normal_shake(self, measures: Measures, k: int) -> tuple[Solution, int | None]:
        solution_to_shake = measures.best_solution
        solution_to_shake_value = measures.best_solution_value

        # Shake and decompose
        shaking_time = time.perf_counter()
        measures.shakes_counter += 1
        x_shake: Solution
        x_shake_value: int | None
        x_shake, x_shake_value = self.shake(
            solution_to_shake,
            solution_to_shake_value,
            self.vns_neighborhood,
            k,
            measures
        )
        shaking_time = time.perf_counter() - shaking_time
        self.logger.debug(f"Shaking time: {str(shaking_time)}")
        measures.cumulative_shaking_time += shaking_time
        return x_shake, x_shake_value

    def build_initial_solutions(self) -> list[tuple[int, ...]]:
        """
        :return: the initial solution
        :rtype: Solution
        """
        items = self.instance.items
        first_items = self.instance.first_items

        # Initialize the list of the families indexes
        families_indexes = list(range(len(first_items)))
        initial_solutions: list[tuple[int, ...]] = []
        
        sorted_families_indexes = sorted(range(len(self.instance.profits)), key=lambda i: self.instance.profits[i], reverse=True)
        solution = self.__build_solution(sorted_families_indexes, first_items, items)
        initial_solutions.append(tuple(solution))

        if len(initial_solutions) == 0:
            # If the families cannot be sorted, then build a solution with random order
            random.shuffle(families_indexes)
            solution = self.__build_solution(families_indexes, first_items, items)
            initial_solutions.append(tuple(solution))
        return initial_solutions
    
    def build_initial_solutions_with_relaxed(self) -> Solution | None:
        """
        :return: the initial solution
        :rtype: Solution
        """
        items = self.instance.items
        first_items = self.instance.first_items

        gurobi_solver = GurobiSolver(
            logger=self.logger
        )
        relaxed_xvars_index_value_dict, reduced_costs_module_index_value_dict, relaxed_obj_value, relaxed_measures = \
            gurobi_solver.solve_relaxed_and_return_families(
            instance=self.instance,
            cpu_time_limit=self.cpu_time_limit,
            output_path=None,
        )

        if relaxed_obj_value is None:
            return None
        
        sorted_families_indexes = sorted(relaxed_xvars_index_value_dict, key=relaxed_xvars_index_value_dict.get, reverse=True) + \
            sorted(reduced_costs_module_index_value_dict, key=reduced_costs_module_index_value_dict.get)

        # Build the solution
        solution = self.__build_solution(sorted_families_indexes, first_items, items)        
        
        return solution

    def __build_solution(self, families_indexes: list[int], first_items: Sequence[int],
                         items: Sequence[Sequence[int]]) -> Solution:
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
        temp_knapsacks1 = dict(sorted(enumerate(self.instance.knapsacks), key=lambda x: sum(x[1]), reverse=True))

        for family_index in families_indexes:  # Iterate through the families
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

        # measures.initial_solution_families_selected.append(selected_families)
        return [value for key, value in sorted(solution.items())]

    def check_stopping_criterion(self, measures: Measures) -> bool:
        """
        Checks if at least one stopping criterion is satisfied.
        :return: True if the stopping criterion is satisfied, False otherwise
        :rtype: bool
        """
        stop = False
        measures.execution_time = time.perf_counter() - measures.start_time
        self.logger.debug(f"Checking stopping criterion")

        if self.instance.optimum is not None and measures.best_solution_value == self.instance.optimum:
            self.finish_measures(measures)
            self.logger.critical("Reached instance's optimum value")
            measures.stopping_cause = Measures.STOP_INSTANCE_OPTIMUM
            stop = True
        elif 0 < self.cpu_time_limit <= measures.execution_time:
            self.finish_measures(measures)
            self.logger.critical("Time limit reached")
            measures.stopping_cause = Measures.STOP_CPU_TIME_LIMIT
            stop = True
        elif 0 < self.iterations_without_improvement_limit <= self.iterations_without_improvement:
            self.logger.critical(f"Iterations without improvement limit reached"
                                 f"\t\t\t{measures.execution_time:.4f}s")
            measures.stopping_cause = Measures.STOP_ITERATIONS_WITHOUT_IMPROVEMENT_LIMIT
            self.finish_measures(measures)
            stop = True
        elif 0 < self.iterations_counter_limit <= measures.iterations_counter:
            self.logger.critical("Iterations limit reached"
                                 f"\t\t\t\t\t\t\t{measures.execution_time:.4f}s")
            measures.stopping_cause = Measures.STOP_ITERATIONS_COUNTER_LIMIT
            self.finish_measures(measures)
            stop = True
        elif self.mip_gap is not None:
            measures.mip_gap = abs(
                (measures.best_solution_value - self.relaxed_optimum_value) / self.relaxed_optimum_value
            )
            self.logger.debug(f"MIPGap: {str(measures.mip_gap)}")
            if measures.mip_gap <= self.mip_gap:
                self.logger.critical(f"MIPGap limit reached"
                                     f"\t\t\t\t\t\t\t\t{measures.execution_time:.4f}s")
                measures.stopping_cause = Measures.STOP_MIP_GAP_LIMIT
                self.finish_measures(measures)
                stop = True
        return stop

    def finish_measures(self, measures: Measures):
        measures.execution_time = time.perf_counter() - measures.start_time
        # Save the number of iterations without improvement of the last solution
        measures.iterations_without_improvement_last = self.iterations_without_improvement

    def shake(self,
              current_solution: Solution,
              current_solution_value: int,
              neighborhood: FamilySwitchNeighborhood,
              k: int,
              measures: Measures
              ) -> tuple[Solution, int | None]:
        """
        Generates a random neighbor of the current solution.
        :param current_solution: the current solution
        :type current_solution: Solution
        :param current_solution_value: the value of the current solution
        :type current_solution_value: int
        :param neighborhood: the neighborhood structure
        :type neighborhood: Neighborhood
        :param k: the number of items to move to generate a neighbor
        :type k: int
        :param measures: the measures of the execution
        :type measures: Measures
        :return: the random neighbor and the list of the indexes of the moved items
        :rtype: tuple[Solution, list[int]]
        """
        self.logger.debug("Shaking")
        neighbor = neighborhood.get_random_neighbor(
            input_solution=current_solution,
            k=k,
            measures=measures
        )
        neighbor_value = None
        if neighbor is None:
            neighbor = current_solution
            neighbor_value = current_solution_value
        return neighbor, neighbor_value
    
    def improve_with_gurobi(self, fixed_families: list[int], solution_to_improve: Solution, solution_to_improve_value: int | None, measures: Measures) -> tuple[Solution, int]:
        model, xvars, yvars, zvars, uvars = build_model(self.instance, disableLogs=True)

        n_items = self.instance.n_items
        n_families = self.instance.n_families
        n_knapsacks = self.instance.n_knapsacks
        first_items = self.instance.first_items
        
        for j, first_item in enumerate(first_items):
            if j in fixed_families:
                xvars[j].LB = xvars[j].UB = 1
                uvars[j].LB = uvars[j].UB = 0
                
                end_item = first_items[l] if (l := j+1) < n_families else n_items
                knapsack = solution_to_improve[first_item]

                for i in range(first_item, end_item):
                    yvars[i, knapsack].LB = yvars[i, knapsack].UB = 1
                    for k in range(n_knapsacks):
                        if k != knapsack:
                            yvars[i, k].LB = yvars[i, k].UB = 0

                zvars[j, knapsack].LB = zvars[j, knapsack].UB = 1
                for k in range(n_knapsacks):
                    if k != knapsack:
                        zvars[j, k].LB = zvars[j, k].UB = 0

                uvars[j].LB = uvars[j].UB = 0
            else:
                xvars[j].LB = xvars[j].UB = 0

        model.update()
        model.optimize()

        best_gurobi_solution = solution_to_improve
        best_gurobi_solution_value = solution_to_improve_value

        if model.solCount > 0:
            best_gurobi_solution = [-1] * self.instance.n_items
            bin_lb = 1 - INT_TOLERANCE
            bin_ub = 1 + INT_TOLERANCE
            for i in range(self.instance.n_items):
                for k in range(self.instance.n_knapsacks):
                    if bin_lb <= yvars[i, k].x <= bin_ub:
                        best_gurobi_solution[i] = k
                        break
            best_gurobi_solution_value = model.objVal

        model.dispose()
        return best_gurobi_solution, best_gurobi_solution_value


    def improve(
            self,
            solution_to_improve: Solution,
            solution_to_improve_value: int | None,
            neighborhood: KRandomRotateNeighborhood,
            variable_indexes: list[int],
            measures: Measures
    ) -> tuple[Solution, int]:
        """
        Improves the solution by trying to move k items to a different knapsack.
        :param solution_to_improve: the solution to improve
        :type solution_to_improve: Solution
        :param solution_to_improve_value: the value of the solution to improve
        :type solution_to_improve_value: int
        :param neighborhood: the neighborhood structure
        :type neighborhood: KRandomRotateNeighborhood
        :param variable_indexes: the indexes of the variables to move
        :type variable_indexes: list[int]
        :param measures: the measures of the execution
        :type measures: Measures
        :return: the improved solution and its objective value
        :rtype: tuple[Solution, int]
        """
        self.logger.debug("Improving - start improving")

        # Initialize the fixed solution and its objective value
        fixed_solution_before_improving = []
        fixed_solution_before_improving_value = 0

        # Init current solution and current objective value
        current_solution = solution_to_improve.copy()
        if solution_to_improve_value is not None:
            current_solution_value = solution_to_improve_value
        else:
            current_solution_value = self.instance.evaluate_solution(current_solution)

        stop = False
        while not stop:  # Cycle executed only once if using best improvement
            k = 1  # Initialize the number of items to move to generate a neighbor
            # Note that this k is different from the k used in the main loop of the VNDS

            measures.vnd_iterations_counter += 1

            # Fix current solution and objective value as the best found so far
            fixed_solution_before_improving = current_solution.copy()
            fixed_solution_before_improving_value = current_solution_value
            self.logger.debug(f"Improving - fixed solution: {str(fixed_solution_before_improving_value)}")

            while k <= len(variable_indexes):  # Search a better solution in every neighborhood
                # Best neighbor near current solution
                get_best_neighbor_time = time.perf_counter()

                best_neighbor, best_neighbor_value = neighborhood.get_best_neighbor(
                    input_solution=current_solution,
                    input_solution_value=current_solution_value,
                    k=k,
                    measures=measures,
                    variable_indexes=variable_indexes
                )
                get_best_neighbor_time = time.perf_counter() - get_best_neighbor_time
                self.logger.debug(f"Improving - get best neighbor time: {str(get_best_neighbor_time)}")
                measures.cumulative_get_best_neighbor_time += get_best_neighbor_time

                self.logger.debug(
                    f"Improving - solution value: {str(fixed_solution_before_improving_value)}")
                self.logger.debug(f"Improving - current solution value: {str(current_solution_value)}")
                self.logger.debug(f"Improving - best neighbor value: {str(best_neighbor_value)}")

                current_solution, current_solution_value, k, has_improved = self.neighborhood_change_step(
                    current_solution, best_neighbor, current_solution_value, best_neighbor_value, k)

                if not has_improved:
                    measures.improvement_failures_counter += 1
                else:
                    measures.improvement_successes_counter += 1

                self.logger.debug(f"Improving - k value: {str(k)}")

            # If no improvement was found then stop
            if fixed_solution_before_improving_value >= current_solution_value:
                self.logger.debug("Improving - no improvement found")
                stop = True

        self.logger.debug(f"Improvement solution value: {str(fixed_solution_before_improving_value)}")
        return fixed_solution_before_improving, fixed_solution_before_improving_value

    def neighborhood_change_step(
            self,
            x_current: Solution,
            x_improvement: Solution,
            current_value: int,
            improvement_value: int,
            k: int
    ) -> tuple[Solution, int, int, bool]:
        """
        Performs the neighborhood change step.
        :param x_current: the current solution
        :type x_current: Solution
        :param x_improvement: the improved solution
        :type x_improvement: Solution
        :param current_value: the value of the current solution
        :type current_value: int
        :param improvement_value: the value of the improved solution
        :type improvement_value: int
        :param k: the number of items to move to generate a neighbor
        :type k: int
        :return: the solution to return, the objective value to return, the new k value and a boolean indicating if the
        solution has improved
        :rtype: tuple[Solution, int, int, bool]
        """
        #
        # Initialize the solution to return, the value to return and the new k value
        solution_to_return = x_current
        value_to_return = current_value
        has_improved = improvement_value > current_value

        if has_improved:
            # If the improved solution is better than the current one, then return it
            # and reset the neighborhood structure to the smallest one
            solution_to_return = x_improvement
            value_to_return = improvement_value
            k = 1
        else:
            # If the improved solution is not better than the current one, then increase the neighborhood structure
            # by increasing the number of items to move
            k += 1

        return solution_to_return, value_to_return, k, has_improved
