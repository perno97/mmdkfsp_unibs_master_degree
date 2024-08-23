from source.data_structures.TopSolution import TopSolution
from source.measures.Measures import Measures



class KernelSearchMeasures:
    KERNEL_SIZE = "kernel_size"
    FAMILIES_PER_BUCKET = "families_per_bucket"
    TOP_SOLUTIONS = "top_solutions"
    TIME_TO_BEST = "TTB"
    ITERATIONS_WITH_ADDED_VARIABLE_COUNTER = "iterations_with_added_variable_counter"
    ADDED_VARIABLES_COUNTER = "added_variables_counter"
    EXECUTION_TIME = "execution_time"
    
    def __init__(self):
        self.start_time: float = 0.0
        self.kernel_size: int = 0
        self.families_per_bucket: int = 0
        self.top_solutions: list[TopSolution] = []
        self.iterations_with_added_variable_counter: int = 0
        self.added_variables_counter: int = 0
        self.best_solution: list[int] = []
        self.best_solution_value: int = 0
        self.execution_time: float = 0.0