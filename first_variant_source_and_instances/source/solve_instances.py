import sys

import logging
import os
from datetime import datetime
from pathlib import Path

from source.solvers.MkfspVnds import *
from mkfsp import load_instance
from source.utils.CustomFormatter import CustomFormatter
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
SKIP_GUROBI = True
GUROBI_TIME_LIMIT = 3600  # Time limit for the Gurobi solver


def solve(parameters: ParametersParser) -> tuple[list[int], list[float]] | None:
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
    init_logging(output_path, logging_level)
    logger = logging.getLogger(__name__)

    # Initialize the results manager
    res_manager = ResultsManager()

    vnds_solutions = []
    vnds_executions = []

    # Solve all the instances
    for path in paths:
        print(f"{'-' * 100}\nSolving instance '{path.name}'")

        instance = load_instance(path)

        # Set logging output path
        if output_path is not None:
            change_logging_path(logger, output_path, instance)

        print(LINES)  # ----------------------------------------
        if ENABLE_PAUSES:
            user_input = input("Type Q to stop or press ENTER to continue\n")
            if user_input.upper() == 'Q':
                return
            print("Starting with VNDS...")

        # Solve with VNDS
        print(f"Solving with VNDS")

        # Create the neighborhoods
        family_switch_neighborhood = FamilySwitchNeighborhood(
            instance=instance,
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
            instance=instance,
            logger=logger,
            items_selection_counter_limit=parameters.item_selection,
            unfeasible_move_items_limit=parameters.move_item,
            use_weights_when_selecting_items=parameters.weights_items,
            use_weights_when_selecting_knapsacks=parameters.weights_knapsacks
        )
        # Create the VNDS solver
        vnds_solver = MkfspVnds(
            k_max=parameters.k_max,
            family_switch_neighborhood=family_switch_neighborhood,
            k_random_rotate_neighborhood=k_random_rotate_neighborhood,
            instance=instance,
            cpu_time_limit=parameters.time,
            iterations_counter_limit=parameters.iterations,
            iterations_without_improvement_limit=parameters.no_improvement,
            logger=logger,
            mip_gap=VNDS_MIP_GAP,
            iterations_without_improvement_before_reset=parameters.restart,
        )

        # Solve the instance
        vnds_solution, vnds_obj_value, vnds_measures = vnds_solver.solve()
        # Check the feasibility of the solution
        vnds_check = instance.check_feasibility(vnds_solution, vnds_obj_value)

        for num, sol in enumerate(vnds_measures.top_solutions):
            if not isinstance(sol, SolutionValueTime):
                raise Exception("Error in top solutions' type")
            print(f"Solution: {num + 1} (value: {-sol.value})")

        # Print the errors returned from the feasibility check, if it's not feasible
        if not vnds_check.is_valid:
            for err_msg in vnds_check.error_messages:
                print(f"Error in feasibility - {err_msg}")
            raise Exception("Error in feasibility")

        print(f"VNDS solver has terminated\n"
                f"Solution: {vnds_obj_value} (valid: {vnds_check.is_valid})")

        res_manager.add_solution(
            instance,
            res_manager.SOLVER_VNDS,
            vnds_obj_value,
            vnds_measures
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
                cpu_time_limit=GUROBI_TIME_LIMIT,
                relaxed=False
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
                if gurobi_solution[0] == vnds_solution:
                    vnds_found_by_gurobi = True

            # Scrivi l'array nel file
            with open(f'{output_path}/vnds_solution_{path.name}.txt', 'w') as f:
                for integer in vnds_solution:
                    f.write(f"{integer},\n")

            with open(f'{output_path}/gurobi_solution_{path.name}.txt', 'w') as f:
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
                if gurobi_solution_value > vnds_obj_value:
                    print(f"\nGurobi solver has found the best solution ({gurobi_solution_value} > {vnds_obj_value})")
                elif gurobi_solution_value < vnds_obj_value:
                    print(f"\nVNDs solver has found the best"
                            f" solution ({vnds_obj_value} > {gurobi_solution_value})")
                else:
                    print(f"\nBoth solvers have found the same best solution ({gurobi_solution_value})")

            res_manager.add_solution(
                instance,
                res_manager.SOLVER_GUROBI,
                int(gurobi_solution_value) if gurobi_solution_value is not None else None,
                gurobi_measures
            )

        # Write results to file        
        res_manager.save_results(output_path)

        # Don't print if evaluating with grid search
        if ENABLE_PAUSES:
            user_input = input("Type Q to stop or press ENTER to continue\n")
            if user_input.upper() == 'Q':
                return
            print("Continuing with the next instance...")

    # Return values for grid search
    return vnds_solutions, vnds_executions


def init_logging(output_path: Path | None, logging_level):
    # Create custom logger logging all five levels
    logger = logging.getLogger(__name__)
    logger.setLevel(logging_level)

    # Define format for logs
    fmt = '%(levelname)4.4s | %(message)s'

    # Create stdout handler for logging to the console (logs all five levels)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(CustomFormatter(fmt))

    # Add both handlers to the logger
    logger.addHandler(stdout_handler)


def change_logging_path(logger: Logger, output_path: Path, instance: Instance):
    for handler in logger.handlers[:]:  # remove all old handlers
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    # Create file handler for logging to a file (logs all five levels)
    logging_folder = Path(output_path, "logs")
    os.makedirs(logging_folder, exist_ok=True)
    file_handler = logging.FileHandler(
        Path(logging_folder, f"{instance.id}.log")
    )
    # Define format for logs
    fmt = '%(levelname)4.4s | %(message)s'
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)  # set the new handler
