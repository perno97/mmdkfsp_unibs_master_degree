from source.data_structures.TopSolution import TopSolution


Solution = list[int]


class Measures:
    # ----------------Following measures are shown in the results--------------------

    # Parameters
    CPU_TIME_LIMIT = "-t"
    ITERATIONS_COUNTER_LIMIT = "-i"
    ITERATIONS_WITHOUT_IMPROVEMENT_LIMIT = "-ni"
    K_MAX = "-k"
    ITEMS_SELECTION_COUNTER_LIMIT = "-is"
    UNFEASIBLE_MOVE_ITEMS_LIMIT = "-mi"
    UNFEASIBLE_ADDITION_LIMIT = "-fa"
    USE_WEIGHTS_WHEN_SELECTING_ITEMS = "-wi"
    USE_WEIGHTS_WHEN_SELECTING_KNAPSACKS = "-wk"
    USE_WEIGHTS_WHEN_ADDING_FAMILIES = "-wa"
    USE_WEIGHTS_WHEN_REMOVING_FAMILIES = "-wr"
    USE_REMOVE_COUNTER_WEIGHTS_REMOVING = "-rwr"
    USE_SELECTION_WEIGHTS_REMOVING = "-swr"
    USE_SELECTION_WEIGHTS_ADDING = "-swa"
    USE_REMOVE_COUNTER_WEIGHTS_ADDING = "-rwa"
    UNFEASIBLE_ADDITION_KNAPSACK_COUNTER_LIMIT = "-ks"
    ITERATIONS_WITHOUT_IMPROVEMENT_BEFORE_RESET = "-r"

    # Main loop
    ITERATIONS_COUNTER = "iterations_counter"
    REACHED_K_MAX_COUNTER = "reached_k_max_counter"
    IMPROVE_TOP_SOLUTION_COUNTER = "-r (restart) counter"
    IMPROVE_TOP_SOLUTION_SUCCESS_COUNTER = "improve_top_solution_success_counter"
    IMPROVEMENTS_COUNTER = "improvements_counter"
    TIME_TO_BEST = "TTB"
    ITERATIONS_TO_BEST = "ITB"
    MAX_SOLUTIONS_DISTANCE = "max_solutions_distance"

    # Build Vnds Measures
    BUILDING_SOLUTION_TIME = "building_solution_time"
    INITIAL_SOLUTION_OBJ_VALUE = "initial_solution_obj_value"

    # Shaking
    MEAN_SHAKING_TIME = "mean_shaking_time"

    # Switch Family Neighborhood Measures
    REMOVAL_NUMBER_VALUES = "removal_number_values"
    UNFEASIBLE_ADD_RANDOM_FAMILIES_COUNTER = "unfeasible_add_random_families_counter"
    NO_FAMILIES_TO_REMOVE_COUNTER = "no_families_to_remove_counter"
    UNFEASIBLE_ADDITION_COUNTER_LIMIT_REACHED = "-fa limit reached"
    UNFEASIBLE_ADDITION_KNAPSACK_COUNTER_LIMIT_REACHED = "-ks limit reached"

    # Improve Vnds Measures
    IMPROVEMENT_FAILURES_COUNTER = "improvement_failures_counter"
    IMPROVEMENT_SUCCESSES_COUNTER = "improvement_successes_counter"
    VND_ITERATIONS_COUNTER = "vnd_iterations_counter"
    NOT_IMPROVEMENTS_COUNTER = "not_improvements_counter"

    # Random Rotate Neighborhood Measures
    DISCARDED_GET_BEST_NEIGHBOR = "discarded_get_best_neighbor"
    MEAN_GET_BEST_NEIGHBOR_TIME = "mean_get_best_neighbor_time"
    UNFEASIBLE_MOVE_ITEMS_LIMIT_REACHED = "-mi limit reached"
    UNFEASIBLE_ITEMS_SELECTION_LIMIT_REACHED = "-is limit reached"    

    # Stopping
    ITERATIONS_WITHOUT_IMPROVEMENT_LAST = "final_iterations_without_improvement"
    VNDS_K_VALUES_COUNTER = "vnds_k_values_counter"
    TOP_SOLUTIONS = "top_solutions"
    STOPPING_CAUSE = "stopping_cause"
    EXECUTION_TIME = "execution_time"

    # --------------------Following measures are NOT shown in the results--------------------

    # Stopping causes
    STOP_CPU_TIME_LIMIT = "stop_cpu_time_limit"
    STOP_ITERATIONS_COUNTER_LIMIT = "stop_iterations_counter_limit"
    STOP_ITERATIONS_WITHOUT_IMPROVEMENT_LIMIT = "stop_iterations_without_improvement_limit"
    STOP_MIP_GAP_LIMIT = "stop_mip_gap_limit"
    STOP_INSTANCE_OPTIMUM = "stop_instance_optimum"
    STOP_GUROBI_OPTIMAL = "stop_gurobi_optimal"
    STOP_GUROBI_INFEASIBLE = "stop_gurobi_infeasible"
    STOP_GUROBI_NODE_LIMIT = "stop_gurobi_node_limit"
    STOP_DEFAULT = "unknown"

    def __init__(self):
        # ----------------Following measures are shown in the results--------------------
        # Constants
        self.vnds_mip_gap: float | None = None
        
        # Parameters
        self.cpu_time_limit: float = 0.0
        self.k_max: int = 0
        self.iterations_without_improvement_limit: int = 0
        self.iterations_counter_limit: int = 0
        self.items_selection_counter_limit: int = 0
        self.unfeasible_move_items_limit: int = 0
        self.use_weights_when_selecting_items: bool = False
        self.use_weights_when_selecting_knapsacks: bool = False
        self.use_weights_when_adding_families: bool = False
        self.use_weights_when_removing_families: bool = False
        self.use_remove_counter_weights_removing: bool = False
        self.use_selection_weights_removing: bool = False
        self.use_selection_weights_adding: bool = False
        self.use_remove_counter_weights_adding: bool = False
        self.unfeasible_addition_limit: int = 0
        self.unfeasible_addition_knapsack_counter_limit: int = 0
        self.iterations_without_improvement_before_reset: int = 0

        # Main loop
        self.iterations_counter: int = 0
        self.reached_kmax_counter: int = 0
        self.improve_top_solution_counter: int = 0
        self.improve_top_solution_success_counter: int = 0
        self.improvements_counter: int = 0
        self.max_solutions_distance: int = 0

        # Build Vnds Measures
        self.building_solution_time: float = 0.0
        self.initial_solution_obj_value: int = 0

        # Switch Family Neighborhood Measures
        self.no_families_to_remove_counter: int = 0
        self.unfeasible_add_random_families: int = 0
        self.removal_number_values: dict[int, int] = {}  # Number of times a removal number was selected
        self.unfeasible_addition_counter_limit_reached: int = 0
        self.unfeasible_addition_knapsack_counter_limit_reached: int = 0

        # Improve Vnds Measures
        self.improvement_failures_counter: int = 0  # Hoy many times best neighbor wasn't better than current solution
        self.improvement_successes_counter: int = 0  # How many times best neighbor was better than current solution
        self.vnd_iterations_counter: int = 0
        self.not_improvements_counter: int = 0

        # Random Rotate Neighborhood Measures
        self.discarded_get_best_neighbor: int = 0
        self.unfeasible_move_items_limit_reached: int = 0
        self.unfeasible_items_selection_limit_reached: int = 0

        # Stopping
        self.iterations_without_improvement_last = 0
        self.vnds_k_values_counter: dict[int, int] = {}  # dict of k values and number of times they were used
        self.top_solutions: list[TopSolution] = []  # list of tuples (solution, objective function value, time)
        self.stopping_cause: str = self.STOP_DEFAULT
        self.best_solution_value: int = 0
        self.execution_time: float = 0.0

        # ----------------Following measures are NOT shown in the results--------------------

        # Main loop
        self.mip_gap: float = 0.0
        self.relaxed_compute_time: float = 0.0
        self.relaxed_optimum_value: int = 0
        self.start_time: float = 0.0
        self.items_selection_score: list[float] = []
        self.items_knapsack_selection_score: list[float] = []
        self.families_selection_score: list[float] = []
        self.families_knapsack_selection_score: list[float] = []
        self.family_mapping: dict[int, int] = {}

        # Shaking
        self.cumulative_shaking_time: float = 0.0
        self.shakes_counter: int = 0

        # Switch Family Neighborhood Measures
        self.families_addition_counter: dict[int, int] = {}  # Family indexes and number of times they were added
        self.removed_families_counter: dict[int, int] = {}  # Family indexes and number of times they were removed
        self.not_selected_families_counter: dict[int, int] = {}  # Number of times they were not selected
        self.selected_families_counter: dict[int, int] = {}  # Number of times they were selected
        self.try_family_add_counter: dict[int, int] = {}  # Number of times a family was tried to be added
        self.add_family_counter: int = 0
        self.remove_counter: int = 0
        # This is also used and updated in Random Rotate Neighborhood
        self.families_knapsacks: dict[int, list[int]] = {}  # dict of families and their knapsacks

        # Improve Vnds Measures
        self.cumulative_improvement_found_time: float = 0.0
        self.cumulative_improvement_not_found_time: float = 0.0

        # Random Rotate Neighborhood Measures
        self.get_best_neighbor_counter: int = 0
        self.cumulative_get_best_neighbor_time: float = 0.0
        self.items_to_move_selected: dict[int, int] = {}  # dict of items and number of times they were selected
        self.knapsack_selection_counter: dict[int, dict[int, int]] = {}  # count item-knapsack selections
        self.item_selection_counter: int = 0
        self.select_knapsack_counter: int = 0

        # Stopping
        self.best_solution: Solution = []

    def increment_family_addition_counter(self, family_index):
        if family_index not in self.families_addition_counter:
            self.families_addition_counter[family_index] = 0
        self.families_addition_counter[family_index] += 1

    def increment_vnds_k_values_counter(self, k):
        if k not in self.vnds_k_values_counter:
            self.vnds_k_values_counter[k] = 0
        self.vnds_k_values_counter[k] += 1

    def increment_selected_families_counter(self, f):
        if f not in self.selected_families_counter:
            self.selected_families_counter[f] = 0
        self.selected_families_counter[f] += 1

    def increment_not_selected_families_counter(self, f):
        if f not in self.not_selected_families_counter:
            self.not_selected_families_counter[f] = 0
        self.not_selected_families_counter[f] += 1

    def increment_removed_families_counter(self, family_index):
        if family_index not in self.removed_families_counter:
            self.removed_families_counter[family_index] = 0
        self.removed_families_counter[family_index] += 1

    def increment_try_family_add_counter(self, f):
        if f not in self.try_family_add_counter:
            self.try_family_add_counter[f] = 0
        self.try_family_add_counter[f] += 1

    def increment_item_to_move_selected(self, i):
        if i not in self.items_to_move_selected:
            self.items_to_move_selected[i] = 0
        self.items_to_move_selected[i] += 1

    def increment_knapsack_selection_counter(self, item: int, knapsack: int):
        if item not in self.knapsack_selection_counter:
            self.knapsack_selection_counter[item] = {}
        if knapsack not in self.knapsack_selection_counter[item]:
            self.knapsack_selection_counter[item][knapsack] = 0
        self.knapsack_selection_counter[item][knapsack] += 1

    def increment_removal_number_values_counter(self, removal_number):
        if removal_number not in self.removal_number_values:
            self.removal_number_values[removal_number] = 0
        self.removal_number_values[removal_number] += 1
