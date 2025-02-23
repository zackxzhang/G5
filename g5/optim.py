from abc import ABC, abstractmethod
from itertools import chain
from math import pi, cos


class Schedule(ABC):

    @abstractmethod
    def step(self):
        pass

    def __call__(self) -> float:
        self.step()
        return self.a  # type: ignore[attr-defined]

    # concatenate schedules by overloading +
    # chain iterators?


class ConstantSchedule(Schedule):

    def __init__(self, a: float):
        self.t = -1
        self.a = a

    def step(self):
        self.t += 1


class LinearSchedule(Schedule):

    def __init__(self, t_max: int, a_max: float, a_min: float):
        self.t = -1
        self.T = t_max
        self.b = a_min
        self.d = (a_max - a_min) / t_max
        self.a = a_max + self.d

    def step(self):
        self.t += 1
        if self.t < self.T:
            self.a -= self.d


class CosineSchedule(Schedule):

    def __init__(self, t_max: int, a_max: float, a_min: float):
        self.t = -1
        self.T = t_max
        self.b = a_min
        self.c = 0.5 * (a_max - a_min)

    def step(self):
        self.t += 1
        if self.t < self.T:
            self.a = self.b + self.c * (1 + cos(pi * self.t / self.T))


class ExponentialSchedule(Schedule):

    def __init__(self, t_max: int, a_max: float, a_min: float, gamma: float):
        self.t = -1
        self.T = t_max
        self.g = gamma
        self.d = 1 / gamma
        self.b = a_min
        self.c = a_max - a_min

    def step(self):
        self.t += 1
        if self.t < self.T:
            self.d *= self.g
            self.a = self.b + self.c * self.d
