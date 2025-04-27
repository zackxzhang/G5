import multiprocessing as mp
import numpy as np                                                # type: ignore
import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from functools import cached_property
from itertools import repeat
from .state import Stone, Board, Coord, Action, onset, proxy, transition, judge
from .agent import Agent


class Permutation:

    def __init__(self, key):
        self.key = key

    def __call__(self, items):
        self.key, subkey = jax.random.split(self.key)
        return jax.random.permutation(subkey, items)


class Rollout:

    """record and columnify"""

    def __init__(self):
        self.frozen = False
        self.coords = list()
        self.rewards = list()
        self.boards = [onset]

    def append(self, coord: Coord, reward: int, board: Board):
        assert not self.frozen
        self.coords.append(coord)
        self.rewards.append(reward)
        self.boards.append(board)

    @property
    def last(self):
        return self.boards[-1]

    @cached_property
    def data(self):
        self.frozen = True
        n = len(self.rewards)
        return (
            jnp.stack([onset] + self.boards[:-2]),
            jnp.stack(self.boards[:-1]),
            jnp.stack(self.coords),
            jnp.stack(self.rewards)[:, None],
            jnp.stack(self.boards[1:]),
            jnp.stack([jnp.nan] * (n-2) + [0.] * 2)[:, None],
            jnp.stack([jnp.nan] * (n-1) + [0.] * 1)[:, None],
        )


headers = [
    'boards_0',
    'boards_1',
    'coords',
    'rewards',
    'boards_2',
    'merits_2',
    'edges',
]


class Collator:

    """concatenate and bisect"""

    def __init__(self, stage: int, batch: int):
        self.frozen = False
        self.prefix = f'stage-{stage}_batch-{batch}_'
        self.p1: dict = {k: list() for k in headers}
        self.p2: dict = {k: list() for k in headers}

    def seed(self, key):
        self.permute = Permutation(key)

    def append(self, *data):
        assert not self.frozen
        for d1, d2, arr in zip(self.p1.values(), self.p2.values(), data):
            d1.append(arr[0::2])
            d2.append(arr[1::2])

    @cached_property
    def data(self):
        self.frozen = True
        v1  = [jnp.vstack(arrays) for arrays in self.p1.values()]
        idx = self.permute(jnp.arange(len(v1[0])))
        p1  = {key: arr[idx] for key, arr in zip(self.p1.keys(), v1)}
        v2  = [jnp.vstack(arrays) for arrays in self.p2.values()]
        idx = self.permute(jnp.arange(len(v2[0])))
        p2  = {key: arr[idx] for key, arr in zip(self.p2.keys(), v2)}
        return p1, p2

    def save(self, f1='p1.npz', f2='p2.npz'):
        p1, p2 = self.data
        np.savez(self.prefix + f1, **{k: np.array(v) for k, v in p1.items()})
        np.savez(self.prefix + f2, **{k: np.array(v) for k, v in p2.items()})


class Loader:

    """shuffle and batch"""

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    def load(self, f1='p1.npz', f2='p2.npz'):
        p1 = np.load(f1)
        p2 = np.load(f2)
        self.p1 = {k: p1[k] for k in headers}
        self.p2 = {k: p2[k] for k in headers}

    def pack(self, data):
        self.p1, self.p2 = data

    @property
    def data(self):
        return self.p1, self.p2

    def __iter__(self):
        pass


class Game:

    def __init__(self, agents: tuple[Agent, Agent]):
        self.agents = agents
        self.round  = 0
        self.winner = 0
        self.rollout = Rollout()

    @property
    def agent(self) -> Agent:
        return self.agents[self.round % 2]

    @property
    def board(self) -> Board:
        return self.rollout.last

    def evo(self, action: Action):
        if self.winner:
            raise ValueError('game over')
        stone, coord = action
        board = transition(self.board, stone, coord)
        winner = judge(board)
        reward = self.agent.eye(winner)
        self.rollout.append(coord, reward, board)
        self.winner = winner
        self.round += 1
        if winner:  # shadow tail of rollout
            self.rollout.append(proxy, self.agent.eye(winner), board)
        return winner


class Score:

    def __init__(self):
        self.wins = [0, 0, 0]

    @property
    def n(self) -> int:
        return sum(self.wins)

    def __call__(self, winner: Stone):
        if winner == 1:
            self.wins[1] += 1
        elif winner == -1:
            self.wins[2] += 1
        else:
            self.wins[0] += 1

    def __str__(self):
        return (
            f'X {self.wins[1]/self.n}\n'
            f'- {self.wins[0]/self.n}\n'
            f'O {self.wins[2]/self.n}\n'
            '--------'
        )


class Simulator:

    def __init__(self, agents: tuple[Agent, Agent]):
        self.agents = agents
        self.score = Score()

    def run(self, stage, division, n_games):
        collator = Collator(stage, batch)
        key = jax.random.key((stage * division + 7) * 11 + 5)
        key, key0, key1 = jax.random.split(key, num=3)
        collator.seed(key)
        self.agents[0].seed(key0)
        self.agents[1].seed(key1)
        for _ in range(n_games):
            game = Game(self.agents)
            while True:
                agent  = game.agent
                action = agent.act(game.board)
                winner = game.evo(action)
                if winner:
                    self.score(winner)
                    break
            collator.append(*game.rollout.data)
        return collator.data

    def __call__(self, stage: int, n_games: int, n_procs=8):
        n = n_games // n_procs
        m = n_games - (n_procs - 1) * n
        with mp.Pool(n_procs) as pool:
            data = pool.starmap(
                self.run,
                zip(repeat(stage), range(n_procs), [n] * (n_procs-1) + [m]),
            )
        return data
