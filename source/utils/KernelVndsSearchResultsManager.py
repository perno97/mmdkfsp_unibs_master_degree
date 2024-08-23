import pandas as pd

from mkfsp.instance import Instance
from source.measures.KernelSearchMeasures import KernelSearchMeasures
from source.measures.Measures import Measures

OBJ_VALUE = "obj_value"
INSTANCE_ID = "instance_id"
SOLVER_NAME = "solver_name"

class KernelVndsSearchResultsManager:    
    """
    Class that manages the saving of results to a pandas dataframe.
    :ivar df_results: dataframe that contains the results
    :ivar SOLVER_VNDS: constant that represents the name of the VNDS solver
    :ivar SOLVER_GUROBI: constant that represents the name of the Gurobi solver
    """

    def __init__(self):
        self.df_results = pd.DataFrame(columns=[
            INSTANCE_ID,
            KernelSearchMeasures.KERNEL_SIZE,
            KernelSearchMeasures.FAMILIES_PER_BUCKET,
            KernelSearchMeasures.TOP_SOLUTIONS,
            KernelSearchMeasures.TIME_TO_BEST,
            KernelSearchMeasures.ITERATIONS_WITH_ADDED_VARIABLE_COUNTER,
            KernelSearchMeasures.ADDED_VARIABLES_COUNTER,
            KernelSearchMeasures.EXECUTION_TIME,
            OBJ_VALUE,
        ])

    def add_solution(self, instance: Instance, obj_value: int | None, kernel_search_measures: Measures):
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

        best_solution = None
        for i in range(1, len(kernel_search_measures.top_solutions) + 1):
            if best_solution is None or kernel_search_measures.top_solutions[-i].value == best_solution.value:
                best_solution = kernel_search_measures.top_solutions[-i]
            if best_solution is not None and kernel_search_measures.top_solutions[-i].value < best_solution.value:
                break

        line_data = [
            instance.id,
            kernel_search_measures.kernel_size,
            kernel_search_measures.families_per_bucket,
            [f"{s.value}-{s.time}s" for s in kernel_search_measures.top_solutions],
            f"{best_solution.time:.5f}",
            kernel_search_measures.iterations_with_added_variable_counter,
            kernel_search_measures.added_variables_counter,
            f"{kernel_search_measures.execution_time:.4f}",
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
