from __future__ import annotations

import json

from collections.abc import Sequence as AbcSequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence as Seq


@dataclass(frozen=True, slots=True)
class FeasibilityCheck:
    """An immutable dataclass for the result of the feasibility analysis."""

    is_valid: bool
    error_messages: Seq[str]


@dataclass(frozen=True, slots=True)
class Instance:
    """An immutable dataclass to describe a MKFSP instance."""

    id: str
    n_items: int
    n_families: int
    n_knapsacks: int
    n_resources: int
    profits: Seq[int]
    penalties: Seq[int]
    first_items: Seq[int]
    items: Seq[Seq[int]]
    knapsacks: Seq[Seq[int]]

    def check_feasibility(self, solution: Seq[int], obj_value: int) -> FeasibilityCheck:
        """Check if the given solution is feasible.

        Args:
            solution (Sequence[int]): a sequence of `n_items` integers.
                `solution[i]` must contain the 0-based index `k` of the
                knapsack where item `i` is loaded or `-1` if item `i` is not
                assigned into any knapsack.
            obj_value (int): the value of the solution.

        Returns:
            FeasibilityCheck: the result of the feasibility check.
        """

        if not isinstance(solution, AbcSequence):
            err_msg = f"Solution must be a sequence, got {type(solution)}"
            return FeasibilityCheck(False, (err_msg,))

        n_items = self.n_items
        n_families = self.n_families
        n_knapsacks = self.n_knapsacks
        n_resources = self.n_resources
        first_items = self.first_items
        profits = self.profits
        penalties = self.penalties
        items = self.items
        knapsacks = self.knapsacks

        if (ls := len(solution)) != n_items:
            err_msg = f"Solution contains {ls} items instead of {n_items}"
            return FeasibilityCheck(False, (err_msg,))

        is_valid = True
        error_messages = []

        expected_obj_value = 0
        splits: Dict[int, set] = {}
        used_resources = [[0] * n_resources for _ in range(n_knapsacks)]
        for j, first_item in enumerate(first_items):
            item_count = 0
            end_item = first_items[l] if (l := j + 1) < n_families else n_items
            for i in range(first_item, end_item):
                k = solution[i]
                if not isinstance(k, int):
                    is_valid = False
                    error_messages.append(
                        f"Value at index {i} is not an integer, got {type(k)}"
                    )
                elif -1 < k < n_knapsacks:
                    item_count += 1
                    for r, resource in enumerate(items[i]):
                        used_resources[k][r] += resource
                    if (s := splits.get(j, None)) is None:
                        splits[j] = s = set()
                    s.add(k)
                elif k != -1:
                    is_valid = False
                    error_messages.append(
                        f"Item {i} is associated to knapsack k = {k} "
                        f"(expected: 0 <= k < {n_knapsacks} or k = -1)"
                    )

            if item_count == (family_size := end_item - first_item):
                expected_obj_value += profits[j] - penalties[j] * (len(splits[j]) - 1)
            elif item_count != 0:
                is_valid = False
                error_messages.append(
                    f"Family {j} is only partially selected: {item_count} "
                    f"items out of {family_size} have been selected"
                )

        for k, resources in enumerate(used_resources):
            for r, used in enumerate(resources):
                if knapsacks[k][r] < used:
                    is_valid = False
                    error_messages.append(
                        f"Knapsack {k} is loaded with {used} units of resource "
                        f"{r} out of a maximum capacity of {knapsacks[k][r]}"
                    )

        if obj_value != expected_obj_value:
            is_valid = False
            error_messages.append(
                f"Objective value {obj_value} != {expected_obj_value} (expected value)"
            )

        return FeasibilityCheck(is_valid, error_messages)

    def is_feasible(self, solution: Seq[int]) -> tuple[bool, list[str]]:
        n_items = self.n_items
        n_families = self.n_families
        n_knapsacks = self.n_knapsacks
        n_resources = self.n_resources
        first_items = self.first_items
        items = self.items
        knapsacks = self.knapsacks

        if (ls := len(solution)) != n_items:
            raise Exception(f"Solution contains {ls} items instead of {n_items}")

        is_valid = True
        error_messages = []
        splits: Dict[int, set] = {}
        used_resources = [[0] * n_resources for _ in range(n_knapsacks)]
        for j, first_item in enumerate(first_items):
            item_count = 0
            end_item = first_items[l] if (l := j + 1) < n_families else n_items
            for i in range(first_item, end_item):
                k = solution[i]
                if not isinstance(k, int):
                    is_valid = False
                    error_messages.append(
                        f"Value at index {i} is not an integer, got {type(k)}"
                    )
                elif -1 < k < n_knapsacks:
                    item_count += 1
                    for r, resource in enumerate(items[i]):
                        used_resources[k][r] += resource
                    if (s := splits.get(j, None)) is None:
                        splits[j] = s = set()
                    s.add(k)
                elif k != -1:
                    is_valid = False
                    error_messages.append(
                        f"Item {i} is associated to knapsack k = {k} "
                        f"(expected: 0 <= k < {n_knapsacks} or k = -1)"
                    )

            if item_count != (family_size := end_item - first_item) and item_count != 0:
                is_valid = False
                error_messages.append(
                    f"Family {j} is only partially selected: {item_count} "
                    f"items out of {family_size} have been selected"
                )

        for k, resources in enumerate(used_resources):
            for r, used in enumerate(resources):
                if knapsacks[k][r] < used:
                    is_valid = False
                    error_messages.append(
                        f"Knapsack {k} is loaded with {used} units of resource "
                        f"{r} out of a maximum capacity of {knapsacks[k][r]}"
                    )

        return is_valid, error_messages

    def evaluate_solution(self, solution) -> int:
        n_families = self.n_families
        first_items = self.first_items
        profits = self.profits
        penalties = self.penalties

        value = 0
        splits: Dict[int, set] = {}
        for j, first_item in enumerate(first_items):  # Iterate through the families
            # Set boundaries for indexing the current family
            end_item = first_items[l] if (l := j + 1) < n_families else self.n_items
            item_count = 0  # Count the number of items of the family that are selected
            for i in range(first_item, end_item):  # Iterate through the items of the family
                k = solution[i]  # Get the knapsack where the item is loaded
                if -1 < k < self.n_knapsacks:  # Check if the item is selected
                    item_count += 1  # Increase the number of selected items
                    if (s := splits.get(j, None)) is None:  # Check if the family is already in the splits
                        splits[j] = s = set()  # If not, add it
                    s.add(k)  # Add the knapsack to the splits
            if item_count == (family_size := end_item - first_item):  # Check if the family is selected
                value += profits[j] - penalties[j] * (len(splits[j]) - 1)  # Increase the value of the solution
        return value


def load_instance(path: Path) -> Instance:
    """Read a MKFSP instance from a JSON file.

    Args:
        path (pathlib.Path): path to the JSON file to be read.

    Returns:
        Instance: the instance.
    """

    with open(path, 'r') as fp:
        data = json.load(fp)
    return Instance(**data)
