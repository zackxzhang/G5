import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from jax import Array                                             # type: ignore
from abc import ABC, abstractmethod
from enum import Flag
from .hint import Stone, Board, Coord, Coords, Action, PyTree
from .state import unravel, affordance, transitions
from .value import Value
from .policy import Policy, critic
from .reward import Reward
from .optim import Schedule, ConstantSchedule
from .codec import encode_key, decode_key, pack_pytree, unpack_pytree


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
    def act(self, board: Board) -> Action:
        pass

    @abstractmethod
    def encode(self) -> PyTree:
        pass

    @classmethod
    def decode(cls, data: PyTree):
        genre = data.pop('class')
        match genre:
            case 'Amateur':
                return Amateur.decode(data)
            case 'ValueLearner':
                return ValueLearner.decode(data)
            case 'PolicyLearner':
                return PolicyLearner.decode(data)
            case _:
                raise ValueError(f"unknown agent class: {genre}")

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
            'class': self.__class__.__name__,
            'stone':  self.stone,
            'reward': self.reward.encode(),
            'key':    encode_key(self._key),
        }

    @classmethod
    def decode(cls, data: PyTree):
        stone  = data['stone']
        reward = Reward.decode(data['reward'])
        key = decode_key(data['key'])
        return cls(stone, reward, key)


class Learner(Agent):

    def __init__(
        self,
        stone: Stone,
        reward: type[Reward],
        epsilon: Schedule,
        key: Array,
    ):
        super().__init__(stone, reward, key)
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

    def __init__(
        self,
        stone: Stone,
        reward: type[Reward],
        value: Value,
        epsilon: Schedule,
        key: Array = jax.random.key(1),
    ):
        super().__init__(stone, reward, epsilon, key)
        self.value = value

    def eval(self):
        self.value.eval()
        return super().eval()

    def encode(self) -> PyTree:
        return {
            'class': self.__class__.__name__,
            'stone':   self.stone,
            'reward':  self.reward.encode(),
            'value':   self.value.encode(),
            'epsilon': self.epsilon.encode(),
            'key':     encode_key(self._key),
        }

    @classmethod
    def decode(cls, data: PyTree):
        stone  = data['stone']
        reward = Reward.decode(data['reward'])
        value  = Value.decode(data['value'])
        epsilon = Schedule.decode(data['epsilon'])
        key = decode_key(data['key'])
        return cls(stone, reward, value, epsilon, key)

    def top(self, board: Board, coords: Coords) -> Coord:
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
        epsilon: Schedule,
        key: Array = jax.random.key(2),
    ):
        super().__init__(stone, reward, epsilon, key)
        self.value = value
        self.policy = policy

    def eval(self):
        self.value.eval()
        self.policy.eval()
        return super().eval()

    def encode(self) -> PyTree:
        return {
            'class': self.__class__.__name__,
            'stone':   self.stone,
            'reward':  self.reward.encode(),
            'value':   self.value.encode(),
            'policy':  self.policy.encode(),
            'epsilon': self.epsilon.encode(),
            'key':     encode_key(self._key),
        }

    @classmethod
    def decode(cls, data: PyTree):
        stone  = data['stone']
        reward = Reward.decode(data['reward'])
        value  = Value.decode(data['value'])
        policy = Policy.decode(data['policy'])
        epsilon = Schedule.decode(data['epsilon'])
        key = decode_key(data['key'])
        return cls(stone, reward, value, policy, epsilon, key)

    def act(self, board: Board) -> Action:
        if self.uniform() < self.epsilon():
            self.mode = Mode.EXPLORE
            coord = self.choice(affordance(board))
        else:
            self.mode = Mode.EXPLOIT
            logpbs = self.policy.predicts(board)
            logpbs = jnp.where(board == 0, logpbs, jnp.nan)
            coord = unravel(int(jnp.nanargmax(logpbs)))
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
        values_0 = self.value(boards_0)
        values_2 = self.value(boards_2)
        advantages = critic(values_0, rewards, values_2, merits_2, edges)
        self.policy.update(boards_1, coords, advantages)
        self.value.update(boards_0, rewards, boards_2, merits_2)
