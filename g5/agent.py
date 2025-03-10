import json
import random
import numpy as np                                                # type: ignore
from abc import ABC, abstractmethod
from enum import Flag
from math import inf
from .state import Stone, Board, Coord, Action, affordance, transitions
from .value import Value
from .policy import Policy, critic
from .reward import Reward
from .optim import Schedule, ConstantSchedule


class Mode(Flag):

    EXPLORE = 1
    EXPLOIT = 2


class Agent(ABC):

    def __init__(self, stone: Stone, reward: type[Reward]):
        self.stone = stone
        self.reward = reward(stone)

    def eye(self, winner: Stone):
        return self.reward(winner)

    @abstractmethod
    def act(self, board: Board) -> Board:
        pass

    @abstractmethod
    def eval(self):
        pass


class Amateur(Agent):

    def __init__(self, stone: Stone, reward: type[Reward]):
        super().__init__(stone, reward)

    def eval(self):
        return self

    def act(self, board: Board) -> Action:
        coords = affordance(board)
        coord = random.choice(list(coords))
        return self.stone, coord


class Learner(Agent):

    def __init__(
        self,
        stone: Stone,
        reward: type[Reward],
        value: Value,
        epsilon: Schedule = ConstantSchedule(0.2),
    ):
        super().__init__(stone, reward)
        self.value = value
        self.epsilon = epsilon
        self.mode: Mode

    def eval(self):
        self.epsilon = ConstantSchedule(0.)
        return self

    @abstractmethod
    def obs(
        self,
        boards_0,
        boards_1,
        coords,
        rewards,
        boards_2,
        merits_2,
        edges,
    ):
        pass


class ValueLearner(Learner):

    def top(self, board: Board, coords: list[Coord]) -> Coord:
        boards = transitions(board, self.stone, coords)
        values = self.value(boards)
        return coords[np.argmax(values)]

    def act(self, board: Board) -> Action:
        coords = affordance(board)
        if random.uniform(0, 1) < self.epsilon():
            self.mode = Mode.EXPLORE
            coord = random.choice(coords)
        else:
            self.mode = Mode.EXPLOIT
            coord = self.top(board, coords)
        return self.stone, coord

    def obs(
        self,
        boards_0,
        boards_1,
        coords,
        rewards,
        boards_2,
        merits_2,
        edges,
    ):
        self.value.update(boards_0, rewards, boards_2, merits_2)


class PolicyLearner(Learner):

    def __init__(
        self,
        stone: Stone,
        reward: type[Reward],
        value: Value,
        policy: Policy,
        epsilon: Schedule = ConstantSchedule(0.2),
    ):
        super().__init__(stone, reward, value, epsilon)
        self.policy = policy

    def act(self, board: Board) -> Action:
        if random.uniform(0, 1) < self.epsilon():
            self.mode = Mode.EXPLORE
            coord = random.choice(affordance(board))
        else:
            self.mode = Mode.EXPLOIT
            logpbs = self.policy(board)
            coord = np.unravel_index(np.argmax(logpbs), shape=logpbs.shape)
        return self.stone, coord

    def obs(
        self,
        boards_0,
        boards_1,
        coords,
        rewards,
        boards_2,
        merits_2,
        edges,
    ):
        advs = critic(self.value, boards_0, rewards, boards_2, merits_2, edges)
        self.policy.update(boards_1, coords, advs)
        self.value.update(boards_0, rewards, boards_2, merits_2)
