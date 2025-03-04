import json
import random
from abc import ABC, abstractmethod
from enum import Flag
from math import inf
from .state import Stone, Board, Action, affordance, transitions
from .value import Value, advantage
from .policy import Policy
from .reward import Reward
from .optim import Schedule, ConstantSchedule


class Mode(Flag):

    EXPLORE = 1
    EXPLOIT = 2


class Agent(ABC):

    def __init__(self, stone: Stone):
        self.stone = stone

    @abstractmethod
    def act(self, board: Board) -> Board:
        pass

    @abstractmethod
    def eval(self):
        pass


class Amateur(Agent):

    def __init__(self, stone: Stone):
        super().__init__(stone)

    def eval(self):
        return self

    def act(self, board: Board) -> Action:
        actions = affordance(board, self.stone)
        action = random.choice(list(actions))
        return action


class Learner(Agent):

    def __init__(
        self,
        stone: Stone,
        value: Value,
        epsilon: Schedule = ConstantSchedule(0.2),
    ):
        super().__init__(stone)
        self.value = value
        self.epsilon = epsilon
        self.mode: Mode

    def eval(self):
        self.epsilon = ConstantSchedule(0.)
        return self

    def top(self, board: Board, actions: list[Action]) -> Action:
        boards = transitions(board, actions)
        values = self.value(boards)
        action_values = list(zip(actions, values))
        random.shuffle(action_values)
        best = -inf
        for a, v in action_values:
            if v > best:
                best, action = v, a
        return action

    def act(self, board: Board) -> Action:
        actions = affordance(board, self.stone)
        if random.uniform(0, 1) < self.epsilon():
            self.mode = Mode.EXPLORE
            action = random.choice(actions)
        else:
            self.mode = Mode.EXPLOIT
            action = self.top(board, actions)
        return action

    @abstractmethod
    def obs(
        self,
        boards_0: list[Board],
        actions: list[Action],
        rewards,
        boards_1: list[Board]
    ):
        pass


class ValueLearner(Learner):

    def obs(
        self,
        boards_0: list[Board],
        actions: list[Action],
        rewards,
        boards_1: list[Board]
    ):
        self.value.update(boards_0, rewards, boards_1)


class PolicyLearner(Learner):

    def __init__(
        self,
        stone: Stone,
        value: Value,
        policy: Policy,
        epsilon: Schedule = ConstantSchedule(0.2),
    ):
        super().__init__(stone, value, epsilon)
        self.policy = policy

    def obs(
        self,
        boards_0: list[Board],
        actions: list[Action],
        rewards,
        boards_1: list[Board]
    ):
        values_0 = self.value(boards_0)
        values_1 = self.value(boards_1)
        advantages = advantage(values_0, rewards, values_1)
        self.policy.update(boards_0, actions, advantages)
        self.value.update(boards_0, rewards, boards_1)
