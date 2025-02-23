from abc import ABC, abstractmethod
from .state import Stone


class Reward(ABC):

    name: str

    def __init__(self, stone: Stone):
        self.stone = stone

    @abstractmethod
    def __call__(self, winner: Stone):
        pass


class Victory(Reward):

    name = 'victory'

    def __call__(self, winner: Stone) -> int:
        if winner == self.stone:
            return +1
        elif winner == -self.stone:
            return -1
        elif winner == 0:
            return 0
        else:
            return 0


class Rush(Reward):

    name = 'rush'

    def __call__(self, winner: Stone) -> int:
        if winner == self.stone:
            return +10
        elif winner == -self.stone:
            return -10
        elif winner == 0:
            return -1
        else:
            return 0


class Survival(Reward):

    name = 'survival'

    def __call__(self, winner: Stone) -> int:
        if winner == self.stone:
            return +10
        elif winner == -self.stone:
            return -10
        elif winner == 0:
            return +1
        else:
            return 0
