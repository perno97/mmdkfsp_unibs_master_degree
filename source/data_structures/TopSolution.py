from typing import Any

Solution = list[int]


class TopSolution:
    """
    Utility class to store a solution and its value, so it can be used in the stack with the top solutions.
    """
    def __init__(self, solution: Solution, value: int, time: float, iteration: int):
        self.solution: Solution = solution
        self.value: int = value
        self.time: float = time
        self.iteration: int = iteration

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, TopSolution):
            raise TypeError(f"Cannot compare TopSolution with {type(other)}")
        return self.value < other.value

    def __neg__(self):
        return TopSolution(self.solution, -self.value, self.time, self.iteration)

    def __str__(self):
        return self.value.__str__()
