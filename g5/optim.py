from abc import ABC, abstractmethod
from itertools import chain
from math import pi, cos


# concatenate schedules by overloading +
# chain iterators?


class Schedule(ABC):

    a: float

    @abstractmethod
    def step(self):
        pass

    def __call__(self) -> float:
        alpha = self.a
        self.step()
        return alpha

    @property
    @abstractmethod
    def params(self) -> dict:
        pass

    def encode(self):
        return {
            'class': self.__class__.__name__,
            'params': self.params,
        }

    @classmethod
    def decode(cls, data):
        match data['class']:
            case 'ConstantSchedule':
                return ConstantSchedule(**data['params'])
            case 'LinearSchedule':
                return LinearSchedule(**data['params'])
            case 'CosineSchedule':
                return CosineSchedule(**data['params'])
            case 'ExponentialSchedule':
                return ExponentialSchedule(**data['params'])
            case _:
                raise ValueError(f"no schedule class named {data['class']}")


class ConstantSchedule(Schedule):

    def __init__(self, a: float, t: int = 0):
        self.t = t
        self.a = a

    def step(self):
        self.t += 1

    @property
    def params(self):
        return {
            'a': self.a,
            't': self.t,
        }


class LinearSchedule(Schedule):

    def __init__(self, t_max: int, a_max: float, a_min: float, t: int = 0):
        self.t = t
        self.t_max = t_max
        self.a_max = a_max
        self.a_min = a_min
        self.d = (a_max - a_min) / t_max
        self.a = a_max - t * self.d

    def step(self):
        self.t += 1
        if self.t < self.t_max:
            self.a -= self.d

    @property
    def params(self):
        return {
            't_max': self.t_max,
            'a_max': self.a_max,
            'a_min': self.a_min,
            't': self.t,
        }


class CosineSchedule(Schedule):

    def __init__(self, t_max: int, a_max: float, a_min: float, t: int = 0):
        self.t_max = t_max
        self.a_max = a_max
        self.a_min = a_min
        self.c = 0.5 * (a_max - a_min)
        self.t = t - 1
        self.step()

    def step(self):
        self.t += 1
        if self.t < self.t_max:
            self.a = self.a_min + self.c * (1 + cos(pi * self.t / self.t_max))

    @property
    def params(self):
        return {
            't_max': self.t_max,
            'a_max': self.a_max,
            'a_min': self.a_min,
            't': self.t,
        }


class ExponentialSchedule(Schedule):

    def __init__(
        self,
        t_max: int,
        a_max: float,
        a_min: float,
        gamma: float,
        t: int = 0,
    ):
        self.t_max = t_max
        self.a_max = a_max
        self.a_min = a_min
        self.c = a_max - a_min
        self.g = gamma
        self.d = gamma ** (t-1)
        self.t = t - 1
        self.step()

    def step(self):
        self.t += 1
        if self.t < self.t_max:
            self.d *= self.g
            self.a = self.a_min + self.c * self.d

    @property
    def params(self):
        return {
            't_max': self.t_max,
            'a_max': self.a_max,
            'a_min': self.a_min,
            'gamma': self.g,
            't': self.t,
        }
