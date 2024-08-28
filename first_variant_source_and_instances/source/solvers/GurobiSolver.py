import os
import time

from gurobipy import GRB

from mkfsp import build_model
from mkfsp.instance import Instance
from pathlib import Path
from logging import Logger

from source.measures.Measures import Measures

INT_TOLERANCE = 1e-6


class GurobiSolver:
    """
    Class that solves the mkfsp problem using Gurobi solver.
    There are three methods: the first solves the integer problem and returns the best solution found; the second solves
    the relaxed problem and returns the best solution found; the third solves the relaxed problem and returns the
    non-null variables indicating the selected families in one dictionary and the reduced costs of the null variables
    in another dictionary.
    """
    def __init__(
            self,
            logger: Logger
    ):
        self.logger: Logger = logger

    def solve(self, instance: Instance, output_path: Path | None, cpu_time_limit: float,
              relaxed: bool = False
              ) -> tuple[list[tuple[list, int]], Measures]:
        """
        Solves the instance using Gurobi solver.
        :return: A tuple containing a list of solutions found and a Measures object, referred to the execution.
        :rtype: tuple[list[tuple[list, int]], Measures]
        """

        measures = Measures()
        measures.start_time = time.perf_counter()
        measures.cpu_time_limit = cpu_time_limit

        model, _, yvars, _, _ = build_model(instance)

        if output_path is not None:
            logging_folder = Path(output_path, 'logs')
            os.makedirs(logging_folder, exist_ok=True)
            model.Params.logFile = str(Path(logging_folder, f'{instance.id}.log'))
        model.Params.timeLimit = cpu_time_limit
        model.Params.MIPGap = 0.0
        if relaxed:
            model.relax()
        model.optimize()
        measures.execution_time = time.perf_counter() - measures.start_time
        if output_path is not None:
            json_folder = Path(output_path, 'gurobi_json_files')
            os.makedirs(json_folder, exist_ok=True)
            model.write(str(Path(json_folder, f'{instance.id}.json')))

        solutions: list[tuple[list, int]] = []
        if model.solCount > 0:
            solution = [-1] * instance.n_items
            bin_lb = 1 - INT_TOLERANCE
            bin_ub = 1 + INT_TOLERANCE
            for i in range(instance.n_items):
                for k in range(instance.n_knapsacks):
                    if bin_lb <= yvars[i, k].x <= bin_ub:
                        solution[i] = k
                        break
            obj_value = model.objVal
            solutions.append((solution, obj_value))

        match model.status:
            case GRB.Status.OPTIMAL:
                self.logger.debug(
                    "The model was solved to optimality")
                measures.stopping_cause = Measures.STOP_GUROBI_OPTIMAL
            case GRB.Status.INFEASIBLE:
                self.logger.debug("The model is infeasible.")
                measures.stopping_cause = Measures.STOP_GUROBI_INFEASIBLE
            case GRB.Status.ITERATION_LIMIT:
                self.logger.debug(
                    "The optimization terminated because the total number of simplex"
                    " iterations performed exceeded the value specified by the IterationLimit parameter.")
                measures.stopping_cause = Measures.STOP_ITERATIONS_COUNTER_LIMIT
            case GRB.Status.NODE_LIMIT:
                self.logger.debug(
                    "The optimization terminated because the total number of branch-and-cut"
                    " nodes explored exceeded the value specified by the NodeLimit parameter.")
                measures.stopping_cause = Measures.STOP_GUROBI_NODE_LIMIT
            case GRB.Status.TIME_LIMIT:
                self.logger.debug(
                    "The optimization terminated because the time expended exceeded"
                    " the value specified by the TimeLimit parameter.")
                measures.stopping_cause = Measures.STOP_CPU_TIME_LIMIT
            case _:
                self.logger.debug(f"Gurobi status code: {str(model.status)}")
                measures.stopping_cause = f"Gurobi status code - {model.status}"

        model.dispose()
        return solutions, measures
    
    def solve_relaxed_and_return_families(self, instance: Instance, output_path: Path | None, cpu_time_limit: float) -> tuple[list[float], list[float]]:
        model, xvars, yvars, zvars, uvars = build_model(instance)
        
        measures = Measures()
        if output_path is not None:
            logging_folder = Path(output_path, 'logs')
            os.makedirs(logging_folder, exist_ok=True)
            model.Params.logFile = str(Path(logging_folder, f'{instance.id}.log'))
        model.Params.timeLimit = cpu_time_limit
        model = model.relax()
        model.optimize()
        
        if output_path is not None:
            json_folder = Path(output_path, 'gurobi_json_files')
            os.makedirs(json_folder, exist_ok=True)
            model.write(str(Path(json_folder, f'{instance.id}.json')))

        best_obj_value = None
        if model.solCount > 0:
            best_obj_value = model.objVal

        match model.status:
            case GRB.Status.OPTIMAL:
                self.logger.debug(
                    "The model was solved to optimality")
            case GRB.Status.INFEASIBLE:
                self.logger.debug("The model is infeasible.")
            case GRB.Status.ITERATION_LIMIT:
                self.logger.debug(
                    "The optimization terminated because the total number of simplex"
                    " iterations performed exceeded the value specified by the IterationLimit parameter.")
            case GRB.Status.NODE_LIMIT:
                self.logger.debug(
                    "The optimization terminated because the total number of branch-and-cut"
                    " nodes explored exceeded the value specified by the NodeLimit parameter.")
            case GRB.Status.TIME_LIMIT:
                self.logger.debug(
                    "The optimization terminated because the time expended exceeded"
                    " the value specified by the TimeLimit parameter.")
            case _:
                self.logger.debug(f"Gurobi status code: {str(model.status)}")
        
        xvars_index_value_dict = {}
        reduced_costs_module_index_value_dict = {}
        for j in range(instance.n_families):
            x = model.getVarByName(f'x[{j}]')
            if x.x > 0.0:
                xvars_index_value_dict[j] = model.getVarByName(f'x[{j}]').x
            else:
                reduced_costs_module_index_value_dict[j] = abs(model.getVarByName(f'x[{j}]').RC)

        model.dispose()
        return xvars_index_value_dict, reduced_costs_module_index_value_dict, best_obj_value, measures
    
    def solve_relaxed(self, instance: Instance, output_path: Path | None, cpu_time_limit: float) -> tuple[int | None, Measures]:
        measures = Measures()
        measures.start_time = time.perf_counter()
        measures.cpu_time_limit = cpu_time_limit

        model, _, _, _, _ = build_model(instance, disable_logs=True)
        model = model.relax()

        if output_path is not None:
            logging_folder = Path(output_path, 'logs')
            os.makedirs(logging_folder, exist_ok=True)
            model.Params.logFile = str(Path(logging_folder, f'{instance.id}.log'))
        model.Params.timeLimit = cpu_time_limit
        model.Params.MIPGap = 0.0
        model.optimize()
        measures.execution_time = time.perf_counter() - measures.start_time
        if output_path is not None:
            json_folder = Path(output_path, 'gurobi_json_files')
            os.makedirs(json_folder, exist_ok=True)
            model.write(str(Path(json_folder, f'{instance.id}.json')))

        best_obj_value = None
        if model.solCount > 0:
            best_obj_value = model.objVal

        match model.status:
            case GRB.Status.OPTIMAL:
                self.logger.debug(
                    "The model was solved to optimality")
                measures.stopping_cause = Measures.STOP_GUROBI_OPTIMAL
            case GRB.Status.INFEASIBLE:
                self.logger.debug("The model is infeasible.")
                measures.stopping_cause = Measures.STOP_GUROBI_INFEASIBLE
            case GRB.Status.ITERATION_LIMIT:
                self.logger.debug(
                    "The optimization terminated because the total number of simplex"
                    " iterations performed exceeded the value specified by the IterationLimit parameter.")
                measures.stopping_cause = Measures.STOP_ITERATIONS_COUNTER_LIMIT
            case GRB.Status.NODE_LIMIT:
                self.logger.debug(
                    "The optimization terminated because the total number of branch-and-cut"
                    " nodes explored exceeded the value specified by the NodeLimit parameter.")
                measures.stopping_cause = Measures.STOP_GUROBI_NODE_LIMIT
            case GRB.Status.TIME_LIMIT:
                self.logger.debug(
                    "The optimization terminated because the time expended exceeded"
                    " the value specified by the TimeLimit parameter.")
                measures.stopping_cause = Measures.STOP_CPU_TIME_LIMIT
            case _:
                self.logger.debug(f"Gurobi status code: {str(model.status)}")
                measures.stopping_cause = f"Gurobi status code - {model.status}"

        model.dispose()
        return best_obj_value, measures