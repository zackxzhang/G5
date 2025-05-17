import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from jax import Array                                             # type: ignore
import numpy as np                                                # type: ignore
from flax.serialization import (                                  # type: ignore
    msgpack_serialize as pack_pytree,
    msgpack_restore as unpack_pytree,
)
from abc import ABC, abstractmethod
from enum import Flag
from math import inf
from .state import Stone, Board, Coord, Action, unravel, affordance, transitions
from .network import PyTree
from .value import Value
from .policy import Policy, critic
from .reward import Reward
from .optim import Schedule, ConstantSchedule


class Mode(Flag):

    EXPLORE = 1
    EXPLOIT = 2


class Agent(ABC):

    def __init__(self, stone: Stone, reward: type[Reward], key: Array):
        self.stone = stone
        self.reward = reward(stone)
        self._key = key

    def eye(self, winner: Stone):
        return self.reward(winner)

    @abstractmethod
    def eval(self):
        pass

    @property
    def key(self):
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def uniform(self):
        return jax.random.uniform(self.key)

    def choice(self, items):
        return jax.random.choice(self.key, items)

    @abstractmethod
    def act(self, board: Board) -> Board:
        pass

    @abstractmethod
    def encode(self) -> PyTree:
        pass

    @classmethod
    @abstractmethod
    def decode(cls, data: PyTree):
        pass

    def save(self, file):
        data = self.encode()
        msgpack = pack_pytree(data)
        with open(file, 'wb') as f:
            f.write(msgpack)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            msgpack = f.read()
        data = unpack_pytree(msgpack)
        return cls.decode(data)


class Amateur(Agent):

    def __init__(
        self,
        stone: Stone,
        reward: type[Reward],
        key: Array = jax.random.key(0),
    ):
        super().__init__(stone, reward, key)

    def eval(self):
        return self

    def act(self, board: Board) -> Action:
        coords = affordance(board)
        coord = self.choice(coords)
        return self.stone, coord

    def encode(self) -> PyTree:
        return {
            'stone':  self.stone,
            'reward': self.reward.encode(),
            'key':    jax.random.key_data(self._key),
        }

    @classmethod
    def decode(cls, data: PyTree):
        stone  = data['stone']
        reward = Reward.decode(data['reward'])
        key = jax.random.wrap_key_data(data['key'])
        return cls(stone, reward, key)


class Learner(Agent):

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

    def __init__(
        self,
        stone: Stone,
        reward: type[Reward],
        value: Value,
        epsilon: Schedule = ConstantSchedule(0.2),
        key: Array = jax.random.key(1),
    ):
        super().__init__(stone, reward, key)
        self.value = value
        self.epsilon = epsilon
        self.mode: Mode

    def encode(self) -> PyTree:
        return {
            'stone':   self.stone,
            'reward':  self.reward.encode(),
            'value':   self.value.encode(),
            'epsilon': self.epsilon.encode(),
            'key':     jax.random.key_data(self._key),
        }

    @classmethod
    def decode(cls, data: PyTree):
        stone  = data['stone']
        reward = Reward.decode(data['reward'])
        value  = Value.decode(data['value'])
        epsilon = Schedule.decode(data['epsilon'])
        key = jax.random.wrap_key_data(data['key'])
        return cls(stone, reward, value, epsilon, key)

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
        key: Array = jax.random.key(2),
    ):
        super().__init__(stone, reward, key)
        self.value = value
        self.policy = policy
        self.epsilon = epsilon
        self.mode: Mode

    def encode(self) -> PyTree:
        return {
            'stone':   self.stone,
            'reward':  self.reward.encode(),
            'value':   self.value.encode(),
            'policy':  self.policy.encode(),
            'epsilon': self.epsilon.encode(),
            'key':     jax.random.key_data(self._key),
        }

    @classmethod
    def decode(cls, data: PyTree):
        stone  = data['stone']
        reward = Reward.decode(data['reward'])
        value  = Value.decode(data['value'])
        policy = Policy.decode(data['policy'])
        epsilon = Schedule.decode(data['epsilon'])
        key = jax.random.wrap_key_data(data['key'])
        return cls(stone, reward, value, policy, epsilon, key)

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
