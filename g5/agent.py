import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from abc import ABC, abstractmethod
from enum import Flag
from math import inf
from .state import Stone, Board, Coord, Action, unravel, affordance, transitions
from .value import Value
from .policy import Policy, critic
from .reward import Reward
from .optim import Schedule, ConstantSchedule


class Mode(Flag):

    EXPLORE = 1
    EXPLOIT = 2


class Choice:

    def __init__(self, key):
        self.key = key

    def __call__(self, items):
        self.key, subkey = jax.random.split(self.key)
        return jax.random.choice(subkey, items)


class Uniform:

    def __init__(self, key):
        self.key = key

    def __call__(self):
        self.key, subkey = jax.random.split(self.key)
        return jax.random.uniform(subkey)


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

    def __init__(
        self,
        stone: Stone,
        reward: type[Reward],
        seed: int = 0,
    ):
        super().__init__(stone, reward)
        self.seed(jax.random.key(seed))

    def eval(self):
        return self

    def seed(self, key):
        self.choice = Choice(key)

    def act(self, board: Board) -> Action:
        coords = affordance(board)
        coord = self.choice(coords)
        return self.stone, coord


class Learner(Agent):

    def __init__(
        self,
        stone: Stone,
        reward: type[Reward],
        value: Value,
        epsilon: Schedule = ConstantSchedule(0.2),
        seed: int = 1,
    ):
        super().__init__(stone, reward)
        self.value = value
        self.epsilon = epsilon
        self.seed(jax.random.key(seed))
        self.mode: Mode

    def eval(self):
        self.epsilon = ConstantSchedule(0.)
        return self

    def seed(self, key):
        key1, key2 = jax.random.split(key)
        self.uniform = Uniform(key1)
        self.choice = Choice(key2)

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
        return coords[jnp.argmax(values)]

    def act(self, board: Board) -> Action:
        coords = affordance(board)
        if self.uniform() < self.epsilon():
            self.mode = Mode.EXPLORE
            coord = self.choice(coords)
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
        seed: int = 2,
    ):
        super().__init__(stone, reward, value, epsilon)
        self.policy = policy

    def act(self, board: Board) -> Action:
        if self.uniform() < self.epsilon():
            self.mode = Mode.EXPLORE
            coord = self.choice(affordance(board))
        else:
            self.mode = Mode.EXPLOIT
            logpbs = self.policy.predicts(board)
            logpbs = jnp.where(board == 0, logpbs, jnp.nan)
            coord = unravel(jnp.nanargmax(logpbs))
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
        advantages = critic(
            self.value, boards_0, rewards, boards_2, merits_2, edges
        )
        self.policy.update(boards_1, coords, advantages)
        self.value.update(boards_0, rewards, boards_2, merits_2)
