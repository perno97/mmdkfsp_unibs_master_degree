import os
import time

from gurobipy import GRB, quicksum

from mkfsp import build_model
from mkfsp.instance import Instance
from pathlib import Path
from logging import Logger

from source.measures.RelaxedLPMeasures import RelaxedLPMeasures
from source.measures.Measures import Measures
from source.utils.RelaxedSolutionInfo import RelaxedSolutionInfo

INT_TOLERANCE = 1e-6

Solution = list[int]


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

    def solve(self, instance: Instance, output_path: Path | None, cpu_time_limit: float) -> tuple[list[tuple[list, int]], Measures]:
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
    

    def solve_relaxed_and_get_info(self, instance: Instance, output_path: Path | None, cpu_time_limit: float):
        measures = RelaxedLPMeasures()
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
        match model.status:
            case GRB.Status.OPTIMAL:
                best_obj_value = model.objVal
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

        if best_obj_value is not None:
            xvars_dict = {}
            xvars_rc_module_dict = {}
            items_selection_score = []
            families_selection_score = []
            items_knapsack_selection_score = {}
            families_knapsack_selection_score = {}
            for j in range(instance.n_families):
                x = model.getVarByName(f'x[{j}]')
                if x.X > 0.0:
                    xvars_dict[j] = x.X
                    
                else:
                    abs_rc = abs(x.RC)
                    xvars_rc_module_dict[j] = abs_rc

                families_knapsack_selection_score[j] = []
                for k in range(instance.n_knapsacks):
                    z_jk = model.getVarByName(f'z[{j},{k}]')
                    if z_jk.X > 0.0:
                        families_knapsack_selection_score[j].append(z_jk.X * 100)
                    else:
                        families_knapsack_selection_score[j].append(1)

            family_index = -1
            for i in range(instance.n_items):
                if i in instance.first_items:
                    family_index += 1
                    families_selection_score.append(0)
                item_score = 0
                items_knapsack_selection_score[i] = []
                for k in range(instance.n_knapsacks):
                    y_ik = model.getVarByName(f'y[{i},{k}]')
                    if y_ik.X > 0.0:
                        item_score += y_ik.X
                        items_knapsack_selection_score[i].append(y_ik.X)
                        families_selection_score[family_index] += y_ik.X * 10
                    else:
                        families_selection_score[family_index] -= abs(y_ik.RC)
                        items_knapsack_selection_score[i].append(-abs(y_ik.RC))
                items_selection_score.append(item_score)
        else:
            raise Exception(f"Model relaxation not solved. Gurobi status code: {model.status}")

        model.dispose()

        relaxed_info = RelaxedSolutionInfo(
            xvars_dict = xvars_dict,
            xvars_rc_module_dict = xvars_rc_module_dict,
            families_selection_score = families_selection_score,
            families_knapsack_selection_score = families_knapsack_selection_score,
            items_selection_score = items_selection_score,
            items_knapsack_selection_score = items_knapsack_selection_score,
            best_obj_value = best_obj_value,
        )
        return relaxed_info, measures
    
    def solve_relaxed_and_get_splits_vars(self, instance: Instance, output_path: Path | None, cpu_time_limit: float):
        measures = RelaxedLPMeasures()
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
        match model.status:
            case GRB.Status.OPTIMAL:
                best_obj_value = model.objVal
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

        svars_dict = {}
        svars_rc_module_dict = {}

        if best_obj_value is not None:
            for j in range(instance.n_families):
                s_j = model.getVarByName(f's[{j}]')
                if s_j.X > 0.0:
                    svars_dict[j] = s_j.X
                else:
                    abs_rc = abs(s_j.RC)
                    svars_rc_module_dict[j] = abs_rc                
        else:
            raise Exception(f"Model relaxation not solved. Gurobi status code: {model.status}")

        model.dispose()
        return svars_dict, svars_rc_module_dict, best_obj_value, measures
    
    def solve_relaxed(self, instance: Instance, output_path: Path | None, cpu_time_limit: float) -> tuple[int | None, Measures]:
        measures = RelaxedLPMeasures()
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
        match model.status:
            case GRB.Status.OPTIMAL:
                best_obj_value = model.objVal
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
    
    def solve_for_kernel_search(self, instance: Instance, output_path: Path | None, cpu_time_limit: float, input_obj_value: int, bucket_indexes: list[int]) -> tuple[list[tuple[list, int]], Measures]:
        """
        Solves the instance using Gurobi solver.
        :return: A tuple containing a list of solutions found and a Measures object, referred to the execution.
        :rtype: tuple[list[tuple[list, int]], Measures]
        """

        measures = Measures()
        measures.start_time = time.perf_counter()
        measures.cpu_time_limit = cpu_time_limit

        model, xvars, yvars, _, svars = build_model(instance)

        lhs = quicksum([xvars[j] * instance.profits[j] - svars[j] * instance.penalties[j] for j in range(instance.n_families)])
        model.addConstr(lhs, GRB.GREATER_EQUAL, input_obj_value, '_kernel_search_obj_constraint')
        if len(bucket_indexes) > 0:
            buckets_xvars = [xvars[j] for j in bucket_indexes]
            lhs = quicksum(buckets_xvars)
            model.addConstr(lhs, GRB.GREATER_EQUAL, 1, '_kernel_search_select_constraint')
            model.update()
            

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
                self.logger.warn(
                    "The model was solved to optimality")
                measures.stopping_cause = Measures.STOP_GUROBI_OPTIMAL
            case GRB.Status.INFEASIBLE:
                self.logger.warn("The model is infeasible.")
                measures.stopping_cause = Measures.STOP_GUROBI_INFEASIBLE
            case GRB.Status.ITERATION_LIMIT:
                self.logger.warn(
                    "The optimization terminated because the total number of simplex"
                    " iterations performed exceeded the value specified by the IterationLimit parameter.")
                measures.stopping_cause = Measures.STOP_ITERATIONS_COUNTER_LIMIT
            case GRB.Status.NODE_LIMIT:
                self.logger.warn(
                    "The optimization terminated because the total number of branch-and-cut"
                    " nodes explored exceeded the value specified by the NodeLimit parameter.")
                measures.stopping_cause = Measures.STOP_GUROBI_NODE_LIMIT
            case GRB.Status.TIME_LIMIT:
                self.logger.warn(
                    "The optimization terminated because the time expended exceeded"
                    " the value specified by the TimeLimit parameter.")
                measures.stopping_cause = Measures.STOP_CPU_TIME_LIMIT
            case _:
                self.logger.warn(f"Gurobi status code: {str(model.status)}")
                measures.stopping_cause = f"Gurobi status code - {model.status}"

        model.dispose()
        
        return solutions, measures
    
    def improve_with_gurobi(self, selected_families_to_improve: list[int], instance: Instance, solution_to_improve: Solution, solution_to_improve_value: int | None, measures: Measures) -> tuple[Solution, int]:
        model, xvars, yvars, zvars, svars = build_model(instance, disable_logs=True)

        n_items = instance.n_items
        n_families = instance.n_families
        n_knapsacks = instance.n_knapsacks
        first_items = instance.first_items

        for j, first_item in enumerate(first_items):
            if j not in selected_families_to_improve:
                end_item = first_items[l] if (l := j+1) < n_families else n_items
                if solution_to_improve[first_item] == -1:
                    xvars[j].LB = xvars[j].UB = 0
                    svars[j].LB = svars[j].UB = 0
                    for k in range(n_knapsacks):
                        zvars[j, k].LB = zvars[j, k].UB = 0
                    for i in range(first_item, end_item):
                        for k in range(n_knapsacks):
                            yvars[i, k].LB = yvars[i, k].UB = 0
                else:
                    xvars[j].LB = xvars[j].UB = 1
                    knapsacks = set()
                    for i in range(first_item, end_item):
                        for k in range(n_knapsacks):
                            if k == solution_to_improve[i]:
                                yvars[i, k].LB = yvars[i, k].UB = 1
                                knapsacks.add(k)
                            else:
                                yvars[i, k].LB = yvars[i, k].UB = 0
                    for k in range(n_knapsacks):
                        if k in knapsacks:
                            zvars[j, k].LB = zvars[j, k].UB = 1
                        else:
                            zvars[j, k].LB = zvars[j, k].UB = 0
                    svars[j].LB = svars[j].UB = len(knapsacks)-1
            else:
                xvars[j].LB = xvars[j].UB = 1

        model.update()
        model.optimize()

        best_gurobi_solution = solution_to_improve
        best_gurobi_solution_value = solution_to_improve_value

        if model.solCount > 0:
            best_gurobi_solution = [-1] * n_items
            bin_lb = 1 - INT_TOLERANCE
            bin_ub = 1 + INT_TOLERANCE
            for i in range(n_items):
                for k in range(n_knapsacks):
                    if bin_lb <= yvars[i, k].x <= bin_ub:
                        best_gurobi_solution[i] = k
                        break
            best_gurobi_solution_value = round(model.objVal)

        check = instance.check_feasibility(best_gurobi_solution, best_gurobi_solution_value)
        if not check.is_valid:
            raise Exception(f"Solution not valid: {check.error_messages}")
        
        model.dispose()
        return best_gurobi_solution, best_gurobi_solution_value