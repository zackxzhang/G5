import multiprocessing as mp
import numpy as np                                                # type: ignore
import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from .state import Stone, Board, Coord, Action, onset, proxy, transition, judge
from .agent import Agent


class Permutation:

    def __init__(self, key):
        self.key = key

    def __call__(self, items):
        self.key, subkey = jax.random.split(self.key)
        return jax.random.permutation(subkey, items)


class Rollout:

    def __init__(self):
        self.coords = list()
        self.rewards = list()
        self.boards = [onset]

    def append(self, coord: Coord, reward: int, board: Board):
        self.coords.append(coord)
        self.rewards.append(reward)
        self.boards.append(board)

    @property
    def last(self):
        return self.boards[-1]

    @property
    def data(self):
        m = len(self.rewards)
        return (
            jnp.stack([onset] + self.boards[:-2]),
            jnp.stack(self.boards[:-1]),
            jnp.stack(self.coords),
            jnp.stack(self.rewards)[:, None],
            jnp.stack(self.boards[1:]),
            jnp.stack([jnp.nan] * (m-2) + [0.] * 2)[:, None],
            jnp.stack([jnp.nan] * (m-1) + [0.] * 1)[:, None],
        )


header = [
    'boards_0',
    'boards_1',
    'coords',
    'rewards',
    'boards_2',
    'merits_2',
    'edges',
]


class Batcher:

    def __init__(self, seed: int = 7):
        self.seed(jax.random.key(seed))
        self.p1: dict = {head: list() for head in header}
        self.p2: dict = {head: list() for head in header}

    def seed(self, key):
        self.permute = Permutation(key)

    def append(self, *data):
        for d1, d2, arr in zip(self.p1.values(), self.p2.values(), data):
            d1.append(arr[0::2])
            d2.append(arr[1::2])

    def batch(self):
        p1  = [jnp.vstack(data) for data in self.p1.values()]
        idx = self.permute(jnp.arange(len(p1[0])))
        p1  = {key: arr[idx] for key, arr in zip(self.p1.keys(), p1)}
        p2  = [jnp.vstack(data) for data in self.p2.values()]
        idx = self.permute(jnp.arange(len(p2[0])))
        p2  = {key: arr[idx] for key, arr in zip(self.p2.keys(), p2)}
        return p1, p2

    def save(self, f1='p1.npz', f2='p2.npz'):
        p1, p2 = self.batch()
        np.savez(f1, **{k: np.array(v) for k, v in p1.items()})
        np.savez(f2, **{k: np.array(v) for k, v in p2.items()})


class Loader:

    def load(self, f1, f2):
        p1 = np.load(f1)
        p2 = np.load(f2)
        self.p1 = {head: p1[head] for head in header}
        self.p2 = {head: p2[head] for head in header}


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

    def run(self, seed):
        key1, key2 = jax.random.split(jax.random.key(seed))
        self.agents[0].seed(key1)
        self.agents[1].seed(key2)
        game = Game(self.agents)
        while True:
            agent  = game.agent
            action = agent.act(game.board)
            winner = game.evo(action)
            if winner:
                self.score(winner)
                break
        return game.rollout.data

    def __call__(self, n: int):
        batcher = Batcher()
        with mp.Pool(8) as pool:
            data = pool.map(self.run, range(n))
        for d in data:
            batcher.append(*d)
        return batcher
