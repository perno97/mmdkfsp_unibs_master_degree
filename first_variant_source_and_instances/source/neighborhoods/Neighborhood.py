from abc import ABC

from mkfsp.instance import Instance

Solution = list[int]


class Neighborhood(ABC):
    """
    Abstract class for neighborhood generation.
    :ivar instance: Instance of the mkfsp problem.
    """

    def __init__(self, instance: Instance):
        self.instance = instance
