from logging import Logger
import random
import time
from typing import Sequence

from source.data_structures.TopSolution import TopSolution
from source.measures.Measures import Measures
from source.neighborhoods.FamilySwitchNeighborhood import FamilySwitchNeighborhood
from source.neighborhoods.KRandomRotateNeighborhood import KRandomRotateNeighborhood
from mkfsp.instance import Instance
from source.solvers.GurobiSolver import GurobiSolver
from source.solvers.InitialSolutionBuilder import InitialSolutionBuilder

Solution = list[int]  # Defines a solution as sequence of integers


class MkfspVnds:
    """
    Class representing a solver of the mkfsp that uses a VNDS algorithm.
    """

    def __init__(
            self,
            logger: Logger,
            initial_solution_builder: InitialSolutionBuilder,
            k_max: int,
            family_switch_neighborhood: FamilySwitchNeighborhood,
            k_random_rotate_neighborhood: KRandomRotateNeighborhood,
            instance: Instance,
            cpu_time_limit: float,
            iterations_counter_limit: int,
            variable_indexes_number_multiplier: int,
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
        self.variable_indexes_number_multiplier: int = variable_indexes_number_multiplier
        self.vns_neighborhood: FamilySwitchNeighborhood = family_switch_neighborhood
        self.vnd_neighborhood: KRandomRotateNeighborhood = k_random_rotate_neighborhood
        self.instance: Instance = instance
        self.initial_solution_builder = initial_solution_builder
        self.gurobi_solver = None

    def solve(self, input_solution: Solution | None = None,  family_mapping: dict[int, int] = None, input_solution_value: int | None = None, \
            families_selection_score: list[float] | None = None, families_knapsack_selection_score: list[float] | None = None,\
            items_selection_score: list[float] | None = None,\
            items_knapsack_selection_score: dict[list[float]] | None = None) \
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
        measures.items_selection_score = items_selection_score
        measures.items_knapsack_selection_score = items_knapsack_selection_score
        measures.families_selection_score = families_selection_score
        measures.families_knapsack_selection_score = families_knapsack_selection_score
        
        if family_mapping is not None:
            measures.family_mapping = family_mapping
        else:
            for family in range(0, len(self.instance.first_items)):
                measures.family_mapping[family] = family

        # Save parameters
        measures.k_max = self.k_max
        measures.vnds_mip_gap = self.mip_gap
        measures.iterations_without_improvement_limit = self.iterations_without_improvement_limit
        measures.cpu_time_limit = self.cpu_time_limit
        measures.iterations_counter_limit = self.iterations_counter_limit
        measures.iterations_without_improvement_before_reset = self.iterations_without_improvement_before_reset

        # Start measuring execution time
        self.logger.info("Starting VNDS algorithm")
        measures.start_time = time.perf_counter()

        # Compute the relaxed optimum value
        if self.mip_gap is not None:
            measures.relaxed_compute_time = time.perf_counter()
            # Solve the relaxed problem with Gurobi to get an upper bound
            if self.gurobi_solver is None:
                self.gurobi_solver = GurobiSolver(
                    logger=self.logger
                )
            relaxed_obj_value, relaxed_measures = self.gurobi_solver.solve_relaxed(
                instance=self.instance,
                cpu_time_limit=self.cpu_time_limit,
                output_path=None,
            )
            measures.relaxed_compute_time = time.perf_counter() - measures.relaxed_compute_time
            if relaxed_measures.stopping_cause != Measures.STOP_GUROBI_OPTIMAL:
                self.logger.warning("The relaxed problem is not optimal")
                self.mip_gap = None
            else:
                self.logger.info(f"Relaxed solution value: {relaxed_obj_value}"
                                 f"\t\t\t\t\t\t\t{relaxed_measures.execution_time:.4f}s")
                self.relaxed_optimum_value = relaxed_obj_value
                measures.relaxed_optimum_value = self.relaxed_optimum_value

        # Initialize the number of iterations without improvement
        self.iterations_without_improvement = 0

        # Build initial solution
        self.logger.debug("Building initial solution")
        measures.building_solution_time = time.perf_counter()
        # Initialize the best solution and its value
        if input_solution is None:
            if self.gurobi_solver is None:
                self.gurobi_solver = GurobiSolver(
                    logger=self.logger
                )
            measures.best_solution, inner_families_selection_score, inner_families_knapsack_selection_score, inner_items_selection_score,\
                inner_items_knapsack_selection_score, relaxed_best_obj_value, relaxed_measures = self.initial_solution_builder.build_initial_solutions_with_relaxed(self.gurobi_solver)
            if families_selection_score is None:
                measures.families_selection_score = inner_families_selection_score
            if families_knapsack_selection_score is None:
                measures.families_knapsack_selection_score = inner_families_knapsack_selection_score
            if items_selection_score is None:
                measures.items_selection_score = inner_items_selection_score
            if items_knapsack_selection_score is None:
                measures.items_knapsack_selection_score = inner_items_knapsack_selection_score
                
            measures.best_solution_value = self.instance.evaluate_solution(measures.best_solution)
        else:
            measures.best_solution = input_solution
            measures.best_solution_value = input_solution_value
        
        measures.building_solution_time = time.perf_counter() - measures.building_solution_time
        measures.initial_solution_obj_value = measures.best_solution_value

        self.logger.info(f"Initial solution value: {str(measures.best_solution_value)}"
                         f"\t\t\t\t\t\t\t{measures.building_solution_time:.4f}s")

        # Memorize current solution families' knapsacks
        self.__update_families_knapsacks(measures, measures.best_solution)

        current_time = time.perf_counter() - measures.start_time
        measures.top_solutions.append(
            TopSolution(measures.best_solution, measures.best_solution_value, current_time, measures.iterations_counter)
        )

        # Initialize counter
        measures.reached_kmax_counter = 0
        stop = False
        # Main loop
        while not stop:
            self.logger.debug(f"Reached k_max {str(measures.reached_kmax_counter)} times")
            k = 1
            measures.reached_kmax_counter += 1
            while k <= self.k_max and not stop:
                measures.iterations_counter += 1
                self.logger.debug(f"Iteration: {str(measures.iterations_counter)}")

                measures.increment_vnds_k_values_counter(k)

                self.logger.debug(f"VNDS k value: {str(k)}")

                trying_to_improve_top = False

                x_shake = None
                if 0 < self.iterations_without_improvement_before_reset <= self.iterations_without_improvement_from_last_reset_counter:
                    self.iterations_without_improvement_from_last_reset_counter = 0
                    measures.improve_top_solution_counter += 1
                    trying_to_improve_top = True
                    x_shake = measures.best_solution
                    x_shake_value = self.instance.evaluate_solution(x_shake)
                else:
                    x_shake = self.normal_shake(measures, k)
                    x_shake_value = self.instance.evaluate_solution(x_shake)

                # Select k variables to improve (decomposition)
                valid_indexes = [i for i, x in enumerate(x_shake) if x != -1]                


                x_improvement = x_shake
                x_improvement_value = x_shake_value
                improvement_time = 0.0

                # If the solution is empty, skip the improvement phase
                if len(valid_indexes) > 0:
                    variable_indexes_number = k * self.variable_indexes_number_multiplier
                    variable_indexes = random.choices(valid_indexes, k=min(variable_indexes_number, len(valid_indexes)))

                    get_best_neighbor_time = time.perf_counter()
                    improvement_time = time.perf_counter()
                    x_improvement, x_improvement_value = self.improve(
                        x_shake, x_shake_value, self.vnd_neighborhood, variable_indexes, measures
                    )
                    get_best_neighbor_time = time.perf_counter() - get_best_neighbor_time
                    self.logger.debug(f"Improving - get best neighbor time: {str(get_best_neighbor_time)}")
                    measures.cumulative_get_best_neighbor_time += get_best_neighbor_time
                    measures.get_best_neighbor_counter += 1

                    improvement_time = time.perf_counter() - improvement_time

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
                    if trying_to_improve_top:
                        measures.improve_top_solution_success_counter += 1
                    self.iterations_without_improvement = 0
                    self.iterations_without_improvement_from_last_reset_counter = 0
                    self.logger.info(f"New best solution found: {str(measures.best_solution_value)}"
                                     f"\t\t\t\t\t\t\t{time.perf_counter() - measures.start_time:.4f}s")
                    measures.cumulative_improvement_found_time += improvement_time
                    measures.improvements_counter += 1
                    current_time = time.perf_counter() - measures.start_time
                    measures.top_solutions.append(
                        TopSolution(measures.best_solution, measures.best_solution_value, current_time, measures.iterations_counter)
                    )
                    distance_from_last_solution = measures.iterations_counter - measures.top_solutions[-2].iteration
                    if measures.max_solutions_distance < distance_from_last_solution:
                        measures.max_solutions_distance = distance_from_last_solution
                else:
                    self.iterations_without_improvement += 1
                    self.iterations_without_improvement_from_last_reset_counter += 1
                    measures.cumulative_improvement_not_found_time += improvement_time
                    measures.not_improvements_counter += 1
                self.logger.debug(f"Current solution value: {str(measures.best_solution_value)}"
                                 f"\t\t\t\t\t\t\t{time.perf_counter() - measures.start_time:.4f}s")
                self.logger.debug(f"VNDS new k value: {str(k)}")

                trying_to_improve_top = False
            
                stop = self.check_stopping_criterion(measures)
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

        # Shake and decompose
        shaking_time = time.perf_counter()
        measures.shakes_counter += 1
        x_shake: Solution

        x_shake = self.shake(
            self.instance,
            solution_to_shake,
            self.vns_neighborhood,
            k,
            measures
        )

        shaking_time = time.perf_counter() - shaking_time
        self.logger.debug(f"Shaking time: {str(shaking_time)}")
        measures.cumulative_shaking_time += shaking_time
        return x_shake

    def check_stopping_criterion(self, measures: Measures) -> bool:
        """
        Checks if at least one stopping criterion is satisfied.
        :return: True if the stopping criterion is satisfied, False otherwise
        :rtype: bool
        """
        stop = False
        measures.execution_time = time.perf_counter() - measures.start_time
        self.logger.debug(f"Checking stopping criterion")

        if 0 < self.cpu_time_limit <= measures.execution_time:
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
              instance: Instance,
              current_solution: Solution,
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
            instance=instance,
            input_solution=current_solution,
            k=k,
            measures=measures
        )
        if neighbor is None:
            neighbor = current_solution            
        return neighbor

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
                    instance=self.instance,
                    input_solution=current_solution,
                    input_solution_value=current_solution_value,
                    k=k,
                    measures=measures,
                    variable_indexes=variable_indexes
                )
                get_best_neighbor_time = time.perf_counter() - get_best_neighbor_time
                self.logger.debug(f"Improving - get best neighbor time: {str(get_best_neighbor_time)}")
                measures.cumulative_get_best_neighbor_time += get_best_neighbor_time

                if not self.instance.is_feasible(best_neighbor):
                    raise Exception("Improving - best neighbor not feasible")

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