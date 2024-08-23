import time
from source.neighborhoods.FamilySwitchNeighborhood import FamilySwitchNeighborhood
from source.neighborhoods.KRandomRotateNeighborhood import KRandomRotateNeighborhood
from source.solvers.GurobiSolver import GurobiSolver
from source.solvers.InitialSolutionBuilder import InitialSolutionBuilder
from source.solvers.KernelVndsSearch import KernelVndsSearch
from source.solvers.MkfspVnds import MkfspVnds


class SequentialVndsKernel:
    def __init__(self, logger, standard_kernel_logger, vnds_kernel_logger, instance, parameters, vnds_kernel_time_limit, heuristic_time_limit, vnds_mipgap, vnds_variable_indexes_number_multiplier, bucket_time_limit):
        self.initial_solution_builder = InitialSolutionBuilder(instance, vnds_kernel_time_limit)
        self.logger = logger
        self.kernel_search_standard_solver = KernelVndsSearch(
            logger=standard_kernel_logger,
            instance=instance,
            parameters=parameters,
            cpu_time_limit=heuristic_time_limit - vnds_kernel_time_limit,
            bucket_time_limit=bucket_time_limit,
            vnds_variable_indexes_number_multiplier=vnds_variable_indexes_number_multiplier)
        self.kernel_search_vnds_solver = KernelVndsSearch(
            logger=vnds_kernel_logger,
            instance=instance,
            parameters=parameters,
            cpu_time_limit=vnds_kernel_time_limit,
            vnds_variable_indexes_number_multiplier=vnds_variable_indexes_number_multiplier)

    def solve(self, instance, output_path, relaxation_time_limit, families_per_bucket, kernel_size, overlapping):
        start_time = time.perf_counter()
        best_solution = None
        best_solution_value = None
        gurobi_solver = GurobiSolver(logger=self.logger)
        self.logger.warn("Solving the relaxed problem to get informations")
        relaxation_info, relaxation_measures = gurobi_solver.solve_relaxed_and_get_info(instance, None, relaxation_time_limit)
        self.logger.warn(f"Relaxed problem solved"
                            f"\t\t\t\t\t\t\t{time.perf_counter() - start_time:.4f}s")
        best_solution, best_solution_value, vnds_measures = self.kernel_search_vnds_solver.solve(
            kernel_size=kernel_size,
            families_per_bucket=families_per_bucket,
            overlapping=overlapping,
            output_path=output_path,
            standard_kernel_search=False,
            input_solution=best_solution,
            input_relaxation_info=relaxation_info)
        best_solution, best_solution_value, kernel_measures = self.kernel_search_standard_solver.solve(
            kernel_size=kernel_size,
            families_per_bucket=families_per_bucket,
            overlapping=overlapping,
            output_path=output_path,
            standard_kernel_search=True,
            input_solution=best_solution,
            input_relaxation_info=relaxation_info)
        return best_solution, best_solution_value, vnds_measures, kernel_measures
