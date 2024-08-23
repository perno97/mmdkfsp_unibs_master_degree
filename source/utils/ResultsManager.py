import pandas as pd

from mkfsp.instance import Instance
from source.measures.Measures import Measures

OBJ_VALUE = "obj_value"
INSTANCE_ID = "instance_id"
SOLVER_NAME = "solver_name"


class ResultsManager:
    """
    Class that manages the saving of results to a pandas dataframe.
    :ivar df_results: dataframe that contains the results
    :ivar SOLVER_VNDS: constant that represents the name of the VNDS solver
    :ivar SOLVER_GUROBI: constant that represents the name of the Gurobi solver
    """

    def __init__(self):
        """
        Constructor of the class.
        """
        self.SOLVER_VNDS = "vnds"
        self.SOLVER_GUROBI = "gurobi"
        self.df_results = pd.DataFrame(columns=[
            INSTANCE_ID,
            SOLVER_NAME,
            # Parameters
            Measures.CPU_TIME_LIMIT,
            Measures.K_MAX,
            Measures.ITERATIONS_COUNTER_LIMIT,
            Measures.ITERATIONS_WITHOUT_IMPROVEMENT_LIMIT,
            Measures.ITEMS_SELECTION_COUNTER_LIMIT,
            Measures.UNFEASIBLE_MOVE_ITEMS_LIMIT,
            Measures.UNFEASIBLE_ADDITION_LIMIT,
            Measures.UNFEASIBLE_ADDITION_KNAPSACK_COUNTER_LIMIT,
            Measures.ITERATIONS_WITHOUT_IMPROVEMENT_BEFORE_RESET,
            Measures.USE_WEIGHTS_WHEN_SELECTING_ITEMS,
            Measures.USE_WEIGHTS_WHEN_SELECTING_KNAPSACKS,
            Measures.USE_WEIGHTS_WHEN_ADDING_FAMILIES,
            Measures.USE_WEIGHTS_WHEN_REMOVING_FAMILIES,
            Measures.USE_REMOVE_COUNTER_WEIGHTS_ADDING,
            Measures.USE_SELECTION_WEIGHTS_ADDING,
            Measures.USE_SELECTION_WEIGHTS_REMOVING,
            Measures.USE_REMOVE_COUNTER_WEIGHTS_REMOVING,
            Measures.BUILDING_SOLUTION_TIME,
            Measures.ITERATIONS_COUNTER,
            Measures.IMPROVEMENTS_COUNTER,
            Measures.IMPROVE_TOP_SOLUTION_COUNTER,
            Measures.IMPROVE_TOP_SOLUTION_SUCCESS_COUNTER,
            Measures.NOT_IMPROVEMENTS_COUNTER,
            Measures.NO_FAMILIES_TO_REMOVE_COUNTER,
            Measures.UNFEASIBLE_ADD_RANDOM_FAMILIES_COUNTER,
            Measures.REMOVAL_NUMBER_VALUES,
            Measures.MEAN_SHAKING_TIME,
            Measures.UNFEASIBLE_MOVE_ITEMS_LIMIT_REACHED,
            Measures.UNFEASIBLE_ITEMS_SELECTION_LIMIT_REACHED,
            Measures.UNFEASIBLE_ADDITION_COUNTER_LIMIT_REACHED,
            Measures.UNFEASIBLE_ADDITION_KNAPSACK_COUNTER_LIMIT_REACHED,
            Measures.REACHED_K_MAX_COUNTER,
            Measures.DISCARDED_GET_BEST_NEIGHBOR,
            Measures.IMPROVEMENT_FAILURES_COUNTER,
            Measures.IMPROVEMENT_SUCCESSES_COUNTER,
            Measures.VND_ITERATIONS_COUNTER,
            Measures.MEAN_GET_BEST_NEIGHBOR_TIME,
            Measures.ITERATIONS_WITHOUT_IMPROVEMENT_LAST,
            Measures.VNDS_K_VALUES_COUNTER,
            Measures.TOP_SOLUTIONS,
            Measures.TIME_TO_BEST,
            Measures.ITERATIONS_TO_BEST,
            Measures.MAX_SOLUTIONS_DISTANCE,
            Measures.INITIAL_SOLUTION_OBJ_VALUE,
            Measures.STOPPING_CAUSE,
            Measures.EXECUTION_TIME,
            OBJ_VALUE,
        ])

    def add_solution(self, instance: Instance, instance_id: str, solver: str, obj_value: int | None, measures: Measures):
        """
        Add a new solution to the dataframe.
        :param instance: instance of the problem
        :type instance: Instance
        :param solver: name of the solver that has found the solution
        :type solver: str
        :param obj_value: objective value of the solution
        :type obj_value: int
        :param measures: values measured during the execution of the solver
        :type measures: Measures
        :return: None
        """
        if measures.shakes_counter != 0:
            mean_shaking_time = measures.cumulative_shaking_time / measures.shakes_counter
        else:
            mean_shaking_time = 0

        if measures.get_best_neighbor_counter != 0:
            mean_get_best_neighbor_time = \
                (measures.cumulative_get_best_neighbor_time
                 / measures.get_best_neighbor_counter)
        else:
            mean_get_best_neighbor_time = 0

        if obj_value is None:
            obj_value_str = "-"
        else:
            obj_value_str = str(obj_value)

        best_solution = None
        for i in range(1, len(measures.top_solutions) + 1):
            if best_solution is None or measures.top_solutions[-i].value == best_solution.value:
                best_solution = measures.top_solutions[-i]
            if best_solution is not None and measures.top_solutions[-i].value < best_solution.value:
                break

        if solver == self.SOLVER_GUROBI:
            line_data = [
                instance_id,
                solver,
                measures.cpu_time_limit,
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",                
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                measures.stopping_cause,
                f"{measures.execution_time:.4f}",
                obj_value_str,
            ]
        else:
            line_data = [
                instance_id,
                solver,
                measures.cpu_time_limit,
                measures.k_max,
                measures.iterations_counter_limit,
                measures.iterations_without_improvement_limit,
                measures.items_selection_counter_limit,
                measures.unfeasible_move_items_limit,
                measures.unfeasible_addition_limit,
                measures.unfeasible_addition_knapsack_counter_limit,
                measures.iterations_without_improvement_before_reset,
                measures.use_weights_when_selecting_items,
                measures.use_weights_when_selecting_knapsacks,
                measures.use_weights_when_adding_families,
                measures.use_weights_when_removing_families,
                measures.use_remove_counter_weights_adding,
                measures.use_selection_weights_adding,
                measures.use_selection_weights_removing,
                measures.use_remove_counter_weights_removing,
                f"{measures.building_solution_time:.4f}",
                measures.iterations_counter,
                measures.improvements_counter,
                measures.improve_top_solution_counter,
                measures.improve_top_solution_success_counter,
                measures.not_improvements_counter,
                measures.no_families_to_remove_counter,
                measures.unfeasible_add_random_families,
                measures.removal_number_values,
                f"{mean_shaking_time:.4f}",
                measures.unfeasible_move_items_limit_reached,
                measures.unfeasible_items_selection_limit_reached,
                measures.unfeasible_addition_counter_limit_reached,
                measures.unfeasible_addition_knapsack_counter_limit_reached,
                measures.reached_kmax_counter,
                measures.discarded_get_best_neighbor,
                measures.improvement_failures_counter,
                measures.improvement_successes_counter,
                measures.vnd_iterations_counter,
                f"{mean_get_best_neighbor_time:.4f}",
                measures.iterations_without_improvement_last,
                measures.vnds_k_values_counter,
                [f"{s.value}-{s.time:.5f}s-{s.iteration}" for s in measures.top_solutions],
                f"{best_solution.time:.5f}" if len(measures.top_solutions) > 0 else "-",
                best_solution.iteration if len(measures.top_solutions) > 0 else "-",
                measures.max_solutions_distance,
                measures.initial_solution_obj_value,
                measures.stopping_cause,
                f"{measures.execution_time:.4f}",
                obj_value,
            ]
        df_new_line = pd.DataFrame([line_data], columns=self.df_results.columns)
        if self.df_results.empty:
            self.df_results = df_new_line
        else:
            self.df_results = pd.concat([self.df_results, df_new_line], ignore_index=True)

    def print_solution_comparison(self):
        """
        Prints all the rows of the dataframe.
        """
        pd.set_option('display.max_columns', None)
        print(self.df_results)

    def save_results(self, output_path, file_name):
        """
        Writes a csv file with the results.
        :param output_path: path where to save the csv file
        """
        self.df_results.to_csv(f"{output_path}/{file_name}", index=False)
