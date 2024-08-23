from logging import Logger
from pathlib import Path
import time
from typing import Sequence

from mkfsp.instance import Instance
from source.data_structures.TopSolution import TopSolution
from source.measures.KernelSearchMeasures import KernelSearchMeasures
from source.neighborhoods.FamilySwitchNeighborhood import FamilySwitchNeighborhood
from source.neighborhoods.KRandomRotateNeighborhood import KRandomRotateNeighborhood
from source.solvers.MkfspVnds import MkfspVnds
from source.solvers.GurobiSolver import GurobiSolver
from source.solvers.InitialSolutionBuilder import InitialSolutionBuilder
from source.utils.ParametersParser import ParametersParser
from source.utils.ResultsManager import ResultsManager


Solution = Sequence[int]

class KernelVndsSearch:
    """
    Class that implements the Kernel Search algorithm with the VNDS solver.
    """
    def __init__(
            self,
            logger: Logger,
            instance: Instance,
            parameters: ParametersParser,
            cpu_time_limit: float,
            vnds_variable_indexes_number_multiplier: float,
            vnds_mipgap: float | None = None,
            bucket_time_limit: float = 0.0,
            ):
        self.logger: Logger = logger
        self.instance: Instance = instance
        self.parameters: ParametersParser = parameters
        self.relaxation_cpu_time_limit: float = cpu_time_limit
        self.cpu_time_limit: float = cpu_time_limit
        self.initial_solution_builder = InitialSolutionBuilder(instance, parameters.time)
        self.vnds_solver = self.__create_vnds_solver(instance, self.initial_solution_builder, logger, parameters,\
                                                     vnds_mipgap, vnds_variable_indexes_number_multiplier)
        self.bucket_time_limit = bucket_time_limit
        self.relaxation_info = None

    def __create_vnds_solver(self, instance, initial_solution_builder,
     logger, parameters, vnds_mipgap, vnds_variable_indexes_number_multiplier):
        # Create the neighborhoods
        family_switch_neighborhood = FamilySwitchNeighborhood(
            logger=logger,
            unfeasible_addition_limit=parameters.family_addition,
            unfeasible_addition_knapsack_counter_limit=parameters.knapsack_selection,
            use_weights_when_removing_families=parameters.weights_removing,
            use_weights_when_adding_families=parameters.weights_adding,
            use_weights_when_selecting_knapsacks=parameters.weights_knapsacks,
            use_remove_counter_weights_removing=parameters.remove_counter_weights_removing,
            use_selection_weights_removing=parameters.selection_counter_weights_removing,
            use_selection_weights_adding=parameters.selection_counter_weights_adding,
            use_remove_counter_weights_adding=parameters.remove_counter_weights_adding
        )
        k_random_rotate_neighborhood = KRandomRotateNeighborhood(
            logger=logger,
            items_selection_counter_limit=parameters.item_selection,
            unfeasible_move_items_limit=parameters.move_item,
            use_weights_when_selecting_items=parameters.weights_items,
            use_weights_when_selecting_knapsacks=parameters.weights_knapsacks
        )
        # Create the VNDS solver
        return MkfspVnds(
            initial_solution_builder=initial_solution_builder,
            k_max=parameters.k_max,
            family_switch_neighborhood=family_switch_neighborhood,
            k_random_rotate_neighborhood=k_random_rotate_neighborhood,
            instance=instance,
            cpu_time_limit=parameters.time,
            iterations_counter_limit=parameters.iterations,
            variable_indexes_number_multiplier=vnds_variable_indexes_number_multiplier,
            iterations_without_improvement_limit=parameters.no_improvement,
            logger=logger,
            mip_gap=vnds_mipgap,
            iterations_without_improvement_before_reset=parameters.restart,
        )

    def solve(self, kernel_size: int, families_per_bucket: int, overlapping: int, output_path: Path, standard_kernel_search: bool, \
              input_solution: Solution | None = None, input_relaxation_info: int | None = None) -> tuple[
                Solution,
                int
            ]:
        self.relaxation_info = input_relaxation_info
        kernel_measures = KernelSearchMeasures()
        vnds_results_manager = ResultsManager()
        kernel_measures.start_time = time.perf_counter()
        kernel_measures.families_per_bucket = families_per_bucket
        self.logger.warn(f"Starting Kernel Search with {kernel_measures.families_per_bucket} families per bucket")

        self.logger.warn("Building Kernel Search initial solution")
        if input_solution is None:
            kernel, buckets = self.initialization(kernel_size, families_per_bucket, overlapping, kernel_measures)
            best_solution, best_solution_value = self.build_initial_solution(kernel)
        else:
            best_solution = input_solution
            best_solution_value = self.instance.evaluate_solution(input_solution)
            kernel, buckets = self.initialization_with_input(input_solution, families_per_bucket, overlapping, kernel_measures)
        kernel_measures.top_solutions.append(
            TopSolution(best_solution, best_solution_value, time.perf_counter() - kernel_measures.start_time, 0)
        )
        relaxed_optimum = self.relaxation_info.best_obj_value
        families_selection_score = self.relaxation_info.families_selection_score
        families_knapsack_selection_score = self.relaxation_info.families_knapsack_selection_score
        items_selection_score = self.relaxation_info.items_selection_score
        items_knapsack_selection_score = self.relaxation_info.items_knapsack_selection_score

        self.logger.warn(f"Kernel size: {len(kernel)}")
        kernel_measures.kernel_size = len(kernel)
        self.logger.warn(f"Relaxed optimum: {relaxed_optimum}"
                            f"\t\t\t\t\t\t\t{time.perf_counter() - kernel_measures.start_time:.4f}s")

        self.logger.warn("Kernel: " + str(kernel))
        self.logger.debug("Selected families --> " + str(self.get_selected_families(best_solution, self.instance.first_items)))
        buckets.insert(0, [])

        bucket_number=0
        for b in buckets:
            bucket_number += 1
            current_time = time.perf_counter() - kernel_measures.start_time
            if current_time > self.cpu_time_limit:
                self.logger.error("Kernel Time limit reached")
                break
            self.logger.warn(f"Solving with bucket number {bucket_number}"
                                f"\t\t\t\t\t\t\t{current_time:.4f}s")
            self.logger.warn("Kernel: " + str(kernel)
                             + f"\t{current_time:.4f}s")
            self.logger.warn("Bucket: " + str(b)
                                + f"\t{current_time:.4f}s")   

            solve_bucket_start_time = time.perf_counter() - kernel_measures.start_time
            current_solution, current_solution_value, vnds_measures = self.solve_bucket(output_path, kernel, b, bucket_number,\
                best_solution, best_solution_value, families_selection_score, families_knapsack_selection_score, items_selection_score,\
                items_knapsack_selection_score, kernel_measures, standard_kernel_search)
            self.logger.warn(f"Solution value: {current_solution_value}"
                             f"\t\t\t\t\t\t\t\t{time.perf_counter() - kernel_measures.start_time:.4f}s")
            check = self.instance.check_feasibility(current_solution, current_solution_value)
            if not check.is_valid:
                raise Exception(f"Solution is not valid: {check.error_messages}")
            
            vnds_results_manager.add_solution(
                self.instance,
                self.instance.id + f"_bucket{bucket_number}",
                vnds_results_manager.SOLVER_VNDS,
                current_solution_value,
                vnds_measures
            )
            added_bucket_variables = self.get_added_bucket_variables(current_solution, b)
            if current_solution_value >= best_solution_value: # Kernel Search constraints
                if len(added_bucket_variables) > 0:
                    kernel_measures.iterations_with_added_variable_counter += 1
                    kernel_measures.added_variables_counter += len(added_bucket_variables)
                
                for sol in vnds_measures.top_solutions:
                    if sol.value >= best_solution_value:
                        kernel_measures.top_solutions.append(
                            TopSolution(sol.solution, sol.value, solve_bucket_start_time + sol.time, bucket_number)
                        )
                current_time = time.perf_counter() - kernel_measures.start_time
                self.logger.warn("Solution improved")
                self.logger.warn(f"Indexes of the families added to the kernel: {added_bucket_variables}"
                                    f"\t\t\t\t\t{current_time:.4f}s")
                kernel.extend(added_bucket_variables) # Enlarge kernel
                best_solution = current_solution
                best_solution_value = current_solution_value
                if best_solution_value == relaxed_optimum:
                    self.logger.error("Optimum found")
                    break
                    
        kernel_measures.best_solution = best_solution
        kernel_measures.best_solution_value = best_solution_value
        kernel_measures.execution_time = time.perf_counter() - kernel_measures.start_time
        suffix = "_standard_kernel_results.csv" if standard_kernel_search else "_vnds_kernel_results.csv"
        file_name = f"{self.instance.id}" + suffix
        vnds_results_manager.save_results(output_path, file_name)
        return best_solution, best_solution_value, kernel_measures
    
    def build_kernel_and_buckets_from_solution(self, solution: Solution, families_per_bucket: int, overlapping: int, measures: KernelSearchMeasures
                                               ) -> list[int]:
        kernel = []
        remaining_families = []
        for j in range(self.instance.n_families):
            if solution[self.instance.first_items[j]] != -1:
                kernel.append(j)
            else:
                remaining_families.append(j)

        buckets = self.split_into_buckets(remaining_families, families_per_bucket, overlapping, measures)
        return kernel, buckets, measures

    def get_selected_families(self, solution: Solution, first_items) -> list[int]:
        selected_families = []
        for j in range(len(first_items)):
            if solution[first_items[j]] != -1:
                selected_families.append(j)
        return selected_families

    def initialization(self, kernel_size: int, families_per_bucket: int, overlapping: int, measures: KernelSearchMeasures):
        gurobi_solver = GurobiSolver(self.logger)
        
        if self.relaxation_info is None:
            self.relaxation_info, relaxation_measures = \
                gurobi_solver.solve_relaxed_and_get_info(self.instance, None, self.relaxation_cpu_time_limit)
            
        xvars_dict = self.relaxation_info.xvars_dict
        families_selection_score = self.relaxation_info.families_selection_score

        sorted_xvars_indexes = sorted(xvars_dict, key=xvars_dict.get, reverse=True)
        kernel = []
        remaining_xvars = [] # TODO for hybrid kernel search
        for index in sorted_xvars_indexes:
            if xvars_dict[index] == 1:
                kernel.append(index)
            else:
                remaining_xvars.append(index)
        negative_score_family_indexes = [i for i, score in enumerate(families_selection_score) if score < 0]
        sorted_null_xvars_indexes = sorted(negative_score_family_indexes, key=lambda i:families_selection_score[i], reverse=True)

        # if kernel_size <= 0:
        #     kernel = sorted_xvars_indexes
        # else:
        #     kernel = sorted_xvars_indexes[:kernel_size] # TODO for other algorithms

        # remaining_families = sorted_xvars_indexes[kernel_size:] + sorted_null_xvars_indexes

        remaining_families = remaining_xvars + sorted_null_xvars_indexes

        buckets = self.split_into_buckets(remaining_families, families_per_bucket, overlapping, measures)

        return kernel, buckets
    
    def initialization_with_input(self, input_solution: Solution, families_per_bucket: int, overlapping: int, measures: KernelSearchMeasures):
        gurobi_solver = GurobiSolver(self.logger)

        if self.relaxation_info is None:
            xvars_dict, xvars_rc_module_dict, families_selection_score, families_knapsack_selection_score, items_selection_score, items_knapsack_selection_score, relaxed_optimum, relaxation_measures = \
                gurobi_solver.solve_relaxed_and_get_info(self.instance, None, self.relaxation_cpu_time_limit)
            
        xvars_dict = self.relaxation_info.xvars_dict
        families_selection_score = self.relaxation_info.families_selection_score

        sorted_xvars_indexes = sorted(xvars_dict, key=xvars_dict.get, reverse=True)
        negative_score_family_indexes = [i for i, score in enumerate(families_selection_score) if score < 0]
        sorted_null_xvars_indexes = sorted(negative_score_family_indexes, key=lambda i:families_selection_score[i], reverse=True)
        
        kernel = []
        remaining_families = []
        for j in sorted_xvars_indexes + sorted_null_xvars_indexes:
            if input_solution[self.instance.first_items[j]] != -1:
                kernel.append(j)
            else:
                remaining_families.append(j)

        remaining_families = remaining_families + sorted_null_xvars_indexes

        buckets = self.split_into_buckets(remaining_families, families_per_bucket, overlapping, measures)

        return kernel, buckets
    
    def split_into_buckets(self, not_kernel_variables, families_per_bucket, overlapping, measures) -> list[list[int]]:
        buckets = []
        bucket = []
        for i in range(0, len(not_kernel_variables), families_per_bucket - overlapping):
            bucket = not_kernel_variables[i:i + families_per_bucket]
            if len(bucket) == families_per_bucket:
                buckets.append(bucket)
        # for family in not_kernel_variables:
        #     bucket.append(family)
        #     if len(bucket) == families_per_bucket:
        #         buckets.append(bucket)
        #         bucket = []
        # if len(bucket) > 0:
        #     buckets.append(bucket)
        return buckets
    
    def build_initial_solution(self, kernel) -> tuple[
                Solution,
                int
            ]:
        solution = self.initial_solution_builder.build_solution(kernel)
        return solution, self.instance.evaluate_solution(solution)
    
    def solve_bucket(self, output_path, kernel, bucket, bucket_number, best_solution, best_solution_value, families_selection_score,\
                    families_knapsack_selection_score, items_selection_score, items_knapsack_selection_score, kernel_measures, standard_kernel_search)-> tuple[
                Solution,
                int
            ]:
        # Create a reduced version of the instance
        id = self.instance.id + f"_bucket{bucket_number}"
        n_knapsacks = self.instance.n_knapsacks
        n_resources = self.instance.n_resources
        knapsacks = self.instance.knapsacks
        profits = []
        penalties = []
        first_items = []
        items = []
        families_count = 0
        solution_to_improve = []
        family_mapping = {}
        bucket_kernel_search = []
        for j in range(self.instance.n_families):
            if j in kernel or j in bucket:
                family_mapping[families_count] = j
                if j in bucket:
                    bucket_kernel_search.append(families_count)
                families_count += 1
                profits.append(self.instance.profits[j])
                penalties.append(self.instance.penalties[j])
                first_item = self.instance.first_items[j]
                last_item = self.instance.first_items[j + 1] if j + 1 < self.instance.n_families else self.instance.n_items
                first_items.append(len(solution_to_improve))
                items.extend(self.instance.items[first_item:last_item])
                solution_to_improve.extend(best_solution[first_item:last_item])

        assert families_count == len(profits) == len(penalties) == len(first_items)

        n_families = families_count
        reduced_instance = Instance(
            id=id,
            n_items=len(items),
            n_families=n_families,
            n_knapsacks=n_knapsacks,
            n_resources=n_resources,
            profits=profits,
            penalties=penalties,
            first_items=first_items,
            items=items,
            knapsacks=knapsacks
        )
        self.vnds_solver.instance = reduced_instance
        # Use the lower between the vnds time limit and the remaining time, otherwise the solver will
        # exceed the global time limit
        remaining_time = self.cpu_time_limit - (time.perf_counter() - kernel_measures.start_time)
        solution_selected_families_before = self.get_selected_families(best_solution, self.instance.first_items)
        self.logger.debug("Selected families before solving bucket --> " + str(solution_selected_families_before))
        
        gurobi_reduced_obj_value = None
        if standard_kernel_search:
            gurobi_solver = GurobiSolver(self.logger)
            reduced_solutions_tuples, measures = gurobi_solver.solve_for_kernel_search(
                instance=reduced_instance,
                output_path=output_path,
                cpu_time_limit=min(remaining_time, self.bucket_time_limit),
                input_obj_value=best_solution_value,
                bucket_indexes=bucket_kernel_search)
            if len(reduced_solutions_tuples) > 0:
                reduced_solution = reduced_solutions_tuples[0][0]
                gurobi_reduced_obj_value = reduced_solutions_tuples[0][1]
            else:
                reduced_solution = solution_to_improve
        else:
            self.vnds_solver.cpu_time_limit = min(remaining_time, self.parameters.time)
            reduced_solution, _, measures = self.vnds_solver.solve(solution_to_improve, family_mapping,\
                    reduced_instance.evaluate_solution(solution_to_improve), families_selection_score,\
                    families_knapsack_selection_score, items_selection_score, items_knapsack_selection_score)
        
        # Cycle all the families, if a family is not in the reduced instance then add it to the solution with no knapsack
        # Otherwise add the reduced instance's solution
        reduced_family_index = 0
        solution = []
        families_from_bucket = []
        for j in range(self.instance.n_families):
            if j in kernel or j in bucket:
                # This family is in the reduced instance, then add the reduced solution
                first_item = reduced_instance.first_items[reduced_family_index]
                last_item = reduced_instance.first_items[reduced_family_index + 1] if reduced_family_index + 1 <\
                    len(reduced_instance.first_items) else len(reduced_instance.items)
                if j in bucket and reduced_solution[first_item] != -1:
                    families_from_bucket.append(j)
                for i in range(first_item, last_item):
                    solution.append(reduced_solution[i])
                # Increase the index for iterating on the reduced instance's families
                reduced_family_index += 1
            else:
                # This family is not in the reduced instance, then add it to the solution with no knapsack
                first_item = self.instance.first_items[j]
                last_item = self.instance.first_items[j + 1] if j + 1 < len(self.instance.first_items) else len(self.instance.items)
                solution.extend([-1] * (last_item - first_item))

        bucket_obj_value = self.instance.evaluate_solution(solution)

        solution_selected_families_after = self.get_selected_families(solution, self.instance.first_items)
        self.logger.debug("Selected families after solving bucket --> " + str(solution_selected_families_after))
        self.logger.warn("Families from bucket --> " + str(families_from_bucket))
        return solution, bucket_obj_value, measures

    def get_added_bucket_variables(self, current_solution, b) -> list[int]:
        # At least one variable of the bucket b must be selected
        # Solution is greater or equal to the best solution
        added_bucket_variables = []
        for family in b:
            if current_solution[self.instance.first_items[family]] != -1:
                added_bucket_variables.append(family)
        return added_bucket_variables