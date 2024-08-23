import sys

import logging
import os
from datetime import datetime
from pathlib import Path

from mkfsp.instance import FeasibilityCheck
from source.solvers.SequentialVndsKernel import SequentialVndsKernel
from source.solvers.KernelVndsSearch import KernelVndsSearch
from source.solvers.MkfspVnds import *
from mkfsp import load_instance
from source.utils.CustomFormatter import CustomFormatter
from source.utils.KernelVndsSearchResultsManager import KernelVndsSearchResultsManager
from source.utils.ParametersParser import ParametersParser
from source.utils.ResultsManager import ResultsManager

# Get the project directory by calling dirname twice
_project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

LINES = "-" * 30
INT_TOLERANCE = 1e-6

# Parameters
INSTANCES_DIR = Path(_project_dir, "instances")  # Path to the instances directory
OUTPUT_DIR = Path(_project_dir, "output")  # Path to the output directory
ENABLE_PAUSES = False  # Enable pauses between the execution of the solvers and the instances
VNDS_MIP_GAP = None  # MIP gap for the VNDS solver, None for don't solve relaxation
VARIABLE_INDEXES_NUMBER_MULTIPLIER = 8
GUROBI_TIME_LIMIT = 3600  # Time limit for the Gurobi solver
BUCKETS_OVERLAPPING = 0
HEURISTIC_ALGORITHM_TIME_LIMIT = 600
BUCKET_TIME_LIMIT = 500
VNDS_KERNEL_TIME_LIMIT = 60

SEQUENTIAL_VNDS_KERNEL = True
USE_KERNEL_SEARCH = False
STANDARD_KERNEL_SEARCH = False
USE_VNDS = False
SKIP_GUROBI = True


def solve(parameters: ParametersParser) -> tuple[list[int], list[float]] | None:
    """
    Prepare the logging, solve the instances in the instances directory with the given parameters with both the proposed
    VNDS algorithm and the Gurobi solver (if enabled), and save the results in the output directory.
    """
    # Set the logging level
    if parameters.debug:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO

    # Create a unique id for the execution
    execution_id = datetime.utcnow().isoformat().replace(':', '_')

    # Get all the paths of the instances
    paths: list[Path] = []
    for dirpath, _, filenames in os.walk(INSTANCES_DIR):
        for fname in filenames:
            if fname.endswith('.json'):
                paths.append(Path(dirpath, fname))
    paths.sort()

    print(f"Solving {len(paths)} instances with a time limit of {parameters.time} "
            f"seconds (execution id: '{execution_id}')")
    if ENABLE_PAUSES:
        user_input = input("Type Q to stop or press ENTER to continue\n")
        if user_input.upper() == 'Q':
            return
        print("Starting...")

    # Create the output directory
    # Replacing ':' in the execution_id string because Windows doesn't allow it in filenames
    output_path = None
    output_path = Path(OUTPUT_DIR, execution_id.replace(':', '_'))
    os.makedirs(output_path)

    # Initialize logging without output path
    logger = init_logging(logging.getLogger(__name__), output_path, logging_level)

    # Initialize the results manager
    if USE_KERNEL_SEARCH:
        kernel_vnds_search_results_manager = KernelVndsSearchResultsManager()
    if USE_VNDS:
        vnds_results_manager = ResultsManager()
    if SEQUENTIAL_VNDS_KERNEL:
        standard_kernel_logger = init_logging(logging.getLogger(f"{__name__}_standard_kernel_logger"), output_path, logging_level)
        vnds_kernel_logger = init_logging(logging.getLogger(f"{__name__}_vnds_kernel_logger"), output_path, logging_level)
        standard_kernel_results_manager = KernelVndsSearchResultsManager()
        vnds_kernel_results_manager = KernelVndsSearchResultsManager()

    # Solve all the instances
    for path in paths:
        print(f"{'-' * 100}\nSolving instance '{path.name}'")

        instance = load_instance(path)
        

        # Set logging output path
        if output_path is not None:
            change_logging_path(logger, output_path, f"{instance.id}.log")
            if SEQUENTIAL_VNDS_KERNEL:
                change_logging_path(standard_kernel_logger, output_path, f"{instance.id}_standard_kernel.log")
                change_logging_path(vnds_kernel_logger, output_path, f"{instance.id}_vnds_kernel.log")

        print(LINES)  # ----------------------------------------
        if ENABLE_PAUSES:
            user_input = input("Type Q to stop or press ENTER to continue\n")
            if user_input.upper() == 'Q':
                return
            print("Starting with VNDS...")

        if USE_VNDS:
            # Solve with VNDS
            print(f"Solving with VNDS")

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
            initial_solution_builder = InitialSolutionBuilder(
                instance=instance,
                cpu_time_limit=parameters.time,
            )

            # Create the VNDS solver
            vnds_solver = MkfspVnds(
                initial_solution_builder=initial_solution_builder,
                k_max=parameters.k_max,
                family_switch_neighborhood=family_switch_neighborhood,
                k_random_rotate_neighborhood=k_random_rotate_neighborhood,
                instance=instance,
                cpu_time_limit=parameters.time,
                iterations_counter_limit=parameters.iterations,
                variable_indexes_number_multiplier = VARIABLE_INDEXES_NUMBER_MULTIPLIER,
                iterations_without_improvement_limit=parameters.no_improvement,
                logger=logger,
                mip_gap=VNDS_MIP_GAP,
                iterations_without_improvement_before_reset=parameters.restart,
            )
        if USE_KERNEL_SEARCH:
            if STANDARD_KERNEL_SEARCH:
                print(f"Solving with KernelSearch standard")
            else:
                print(f"Solving with KernelSearchVNDS")
            kernel_vnds_search_solver = KernelVndsSearch(
                logger=logger,
                instance=instance,
                parameters=parameters,
                cpu_time_limit=HEURISTIC_ALGORITHM_TIME_LIMIT,
                vnds_mipgap=VNDS_MIP_GAP,
                vnds_variable_indexes_number_multiplier=VARIABLE_INDEXES_NUMBER_MULTIPLIER,
                bucket_time_limit=BUCKET_TIME_LIMIT
            )
        if SEQUENTIAL_VNDS_KERNEL:
            print(f"Solving with SequentialVNDSKernel")
            #vnds_kernel_time_limit = round(0.05 * instance.n_families + 6.4)
            # vnds_kernel_time_limit = round(0.042 * instance.n_families + 57.13)
            # bucket_time_limit = round(0.11 * instance.n_families + 393)
            sequential_vnds_kernel = SequentialVndsKernel(
                logger=logger,
                standard_kernel_logger=standard_kernel_logger,
                vnds_kernel_logger=vnds_kernel_logger,
                instance=instance,
                parameters=parameters,
                bucket_time_limit=BUCKET_TIME_LIMIT,
                heuristic_time_limit=HEURISTIC_ALGORITHM_TIME_LIMIT,
                vnds_kernel_time_limit=VNDS_KERNEL_TIME_LIMIT,
                vnds_mipgap=VNDS_MIP_GAP,
                vnds_variable_indexes_number_multiplier=VARIABLE_INDEXES_NUMBER_MULTIPLIER
            )

        # kernel_size = round(0.07 * instance.n_families)
        # kernel_size = round(0.15 * instance.n_families)
        # families_per_bucket = round(0.06 * instance.n_families)
        #families_per_bucket = round(FAMILIES_PER_BUCKET_PERCENT * instance.n_families)
        # kernel_size = round(0.1 * instance.n_families)
        # families_per_bucket = 7
        # kernel_size = 10
        # families_per_bucket = round(0.05 * instance.n_families + 4)
        #families_per_bucket = round(0.15 * instance.n_families)
        # kernel_size = round(0.005 * instance.n_families + 14.6)
        #families_per_bucket = round(0.1 * instance.n_families)
        kernel_size = 0
        families_per_bucket = round(0.1 * instance.n_families)
        if not USE_VNDS:
            solution = [-1] * instance.n_items
            solution_value = 0
            measures = Measures()
            solution_check = FeasibilityCheck(True, [])
        if USE_KERNEL_SEARCH:
            # Solve the instance
            solution, solution_value, measures = \
                kernel_vnds_search_solver.solve(kernel_size, families_per_bucket, BUCKETS_OVERLAPPING, output_path, STANDARD_KERNEL_SEARCH)
            # Check the feasibility of the solution
            solution_check = instance.check_feasibility(solution, solution_value)
        if USE_VNDS:
            # Solve the instance
            solution, solution_value, measures = vnds_solver.solve()
            # Check the feasibility of the solution
            solution_check = instance.check_feasibility(solution, solution_value)
        if SEQUENTIAL_VNDS_KERNEL:
            solution, solution_value, measures, kernel_measures = sequential_vnds_kernel.solve(instance, output_path, HEURISTIC_ALGORITHM_TIME_LIMIT, families_per_bucket, kernel_size, BUCKETS_OVERLAPPING)

        if SEQUENTIAL_VNDS_KERNEL:
            for num, sol in enumerate(kernel_measures.top_solutions):
                if not isinstance(sol, TopSolution):
                    raise Exception("Error in top solutions' type")
                print(f"Solution: {num + 1} (value: {sol.value})")

        else:
            for num, sol in enumerate(measures.top_solutions):
                if not isinstance(sol, TopSolution):
                    raise Exception("Error in top solutions' type")
                print(f"Solution: {num + 1} (value: {sol.value})")

        # Print the errors returned from the feasibility check, if it's not feasible
        if not solution_check.is_valid:
            for err_msg in solution_check.error_messages:
                print(f"Error in feasibility - {err_msg}")
            raise Exception("Error in feasibility")        

        if USE_KERNEL_SEARCH:
            print(f"KernelSearchVNDS solver has terminated")
        if USE_VNDS:
            print(f"VNDS solver has terminated")
        if SEQUENTIAL_VNDS_KERNEL:
            print(f"SequentialVNDSKernel solver has terminated")
        
        print(f"Solution: {solution_value} (valid: {solution_check.is_valid})")
        
        
        # Write vnds solution to file
        with open(f'{output_path}/my_algorithm_solution_{instance.id}.txt', 'w') as f:
            for integer in solution:
                f.write(f"{integer},\n")
        
        if USE_KERNEL_SEARCH:
            kernel_vnds_search_results_manager.add_solution(
                instance=instance,
                # instance_id=instance.id,
                # solver=kernel_vnds_search_results_manager.SOLVER_VNDS,
                obj_value=solution_value,
                kernel_search_measures=measures
            )
        if USE_VNDS:
            vnds_results_manager.add_solution(
                instance=instance,
                instance_id=instance.id,
                solver=vnds_results_manager.SOLVER_VNDS,
                obj_value=solution_value,
                measures=measures
            )
        if SEQUENTIAL_VNDS_KERNEL:
            vnds_kernel_results_manager.add_solution(
                instance=instance,
                # instance_id=instance.id,
                # solver=kernel_vnds_search_results_manager.SOLVER_VNDS,
                obj_value=solution_value,
                kernel_search_measures=measures
            )
            standard_kernel_results_manager.add_solution(
                instance=instance,
                # instance_id=instance.id,
                # solver=kernel_vnds_search_results_manager.SOLVER_VNDS,
                obj_value=solution_value,
                kernel_search_measures=kernel_measures
            )

        # Gurobi solver part
        if not SKIP_GUROBI:
            print(LINES)  # ----------------------------------------
            if ENABLE_PAUSES:
                user_input = input("Type Q to stop or press ENTER to continue\n")
                if user_input.upper() == 'Q':
                    return
                print("Starting with Gurobi...")

            # Solve with Gurobi
            gurobi_solver = GurobiSolver(logger=logger)

            # Solve the instance
            gurobi_solutions, gurobi_measures = gurobi_solver.solve(
                instance=instance,
                output_path=output_path,
                cpu_time_limit=GUROBI_TIME_LIMIT
            )

            print(f"Gurobi solver has terminated with status code: {gurobi_measures.stopping_cause}\n"
                    f"Gurobi solver found {len(gurobi_solutions)} solutions:")

            # Check the solution found by gurobi
            vnds_found_by_gurobi = False
            gurobi_solution = None
            if len(gurobi_solutions) > 0:
                gurobi_solution = gurobi_solutions[0]
                gurobi_check = instance.check_feasibility(gurobi_solution[0], gurobi_solution[1])

                print(f"Solution: {gurobi_solution[1]} (valid: {gurobi_check.is_valid})\n")

                # Print the errors returned from the feasibility check, if it's not feasible
                if not gurobi_check.is_valid:
                    for err_msg in gurobi_check.error_messages:
                        print(f"  - {err_msg}")

                # Check if the solution found by the Gurobi solver is the same as the one found by the VNDS solver
                if gurobi_solution[0] == solution:
                    vnds_found_by_gurobi = True

            # Write gurobi solution to file
            with open(f'{output_path}/gurobi_solution_{instance.id}.txt', 'w') as f:
                for integer in gurobi_solution:
                    f.write(f"{integer},\n")

            print(LINES)  # ----------------------------------------
            if ENABLE_PAUSES:
                user_input = input("Type Q to stop or press ENTER to continue\n")
                if user_input.upper() == 'Q':
                    return

            # Print if the Gurobi solver has found the same solution as the VNDs solver
            if vnds_found_by_gurobi:
                print(f"\nVNDs found the same solution as Gurobi solver")
            else:
                print(f"\nVNDs didn't find the same solution as Gurobi solver")

            gurobi_solution_value = None
            # Get gurobi's solution value
            if gurobi_solution is not None:
                gurobi_solution_value = gurobi_solution[1]

                # Print which solver has found the best solution and the two values
                if gurobi_solution_value > solution_value:
                    print(f"\nGurobi solver has found the best solution ({gurobi_solution_value} > {solution_value})")
                elif gurobi_solution_value < solution_value:
                    print(f"\nVNDs solver has found the best"
                            f" solution ({solution_value} > {gurobi_solution_value})")
                else:
                    print(f"\nBoth solvers have found the same best solution ({solution_value})")

            if USE_KERNEL_SEARCH:
                kernel_vnds_search_results_manager.add_gurobi_solution(
                    instance,
                    # kernel_vnds_search_results_manager.SOLVER_GUROBI,
                    int(gurobi_solution_value) if gurobi_solution_value is not None else None,
                    gurobi_measures
                )
            else:
                vnds_results_manager.add_solution(
                    instance,
                    instance.id,
                    vnds_results_manager.SOLVER_GUROBI,
                    int(gurobi_solution_value) if gurobi_solution_value is not None else None,
                    gurobi_measures
                )

        # Write results to file
        if SEQUENTIAL_VNDS_KERNEL:
            vnds_kernel_results_manager.save_results(output_path, "vnds_kernel_results.csv")
            standard_kernel_results_manager.save_results(output_path, "standard_kernel_results.csv")
        else:
            file_name = "execution_results.csv"
            if USE_KERNEL_SEARCH:
                kernel_vnds_search_results_manager.save_results(output_path, file_name)
            else:
                vnds_results_manager.save_results(output_path, file_name)

        if ENABLE_PAUSES:
            user_input = input("Type Q to stop or press ENTER to continue\n")
            if user_input.upper() == 'Q':
                return
            print("Continuing with the next instance...")


def init_logging(logger, output_path: Path | None, logging_level):
    # Create custom logger logging all five levels
    logger.setLevel(logging_level)

    # Define format for logs
    fmt = '%(levelname)4.4s | %(message)s'

    # Create stdout handler for logging to the console (logs all five levels)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(CustomFormatter(fmt))

    # Add both handlers to the logger
    logger.addHandler(stdout_handler)
    return logger


def change_logging_path(logger: Logger, output_path: Path, filename: str):
    for handler in logger.handlers[:]:  # remove all old handlers
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    # Create file handler for logging to a file (logs all five levels)
    logging_folder = Path(output_path, "logs")
    os.makedirs(logging_folder, exist_ok=True)
    file_handler = logging.FileHandler(
        Path(logging_folder, filename)
    )
    # Define format for logs
    fmt = '%(levelname)4.4s | %(message)s'
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)  # set the new handler
