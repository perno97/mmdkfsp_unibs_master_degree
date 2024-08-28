from typing import Any

Solution = list[int]


class SolutionValueTime:
    """
    Utility class to store a solution and its value, so it can be used in the stack with the top solutions.
    """
    def __init__(self, solution: Solution, value: int, time: int):
        self.solution: Solution = solution
        self.value: int = value
        self.time: int = time

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, SolutionValueTime):
            raise TypeError(f"Cannot compare SolutionValueTime with {type(other)}")
        return self.value < other.value

    def __neg__(self):
        return SolutionValueTime(self.solution, -self.value)

    def __str__(self):
        return self.value.__str__()
