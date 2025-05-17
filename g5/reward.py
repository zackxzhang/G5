from abc import ABC, abstractmethod
from .state import Stone


class Reward(ABC):

    def __init__(self, stone: Stone):
        self.stone = stone

    @abstractmethod
    def __call__(self, winner: Stone) -> int:
        pass

    def encode(self):
        return {
            'class': self.__class__.__name__,
        }

    @classmethod
    def decode(cls, data):
        match data['class']:
            case 'Victory':
                return Victory
            case 'Rush':
                return Rush
            case 'Survival':
                return Survival
            case _:
                raise ValueError(f"no policy class named {data['class']}")


class Victory(Reward):

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

    def __call__(self, winner: Stone) -> int:
        if winner == self.stone:
            return +10
        elif winner == -self.stone:
            return -10
        elif winner == 0:
            return +1
        else:
            return 0
