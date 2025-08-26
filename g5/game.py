import threading
import numpy as np                                                # type: ignore
import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from jax import Array                                             # type: ignore
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from itertools import repeat
from pathlib import Path
from .hint import Stone, Board, Coord, Action
from .state import onset, proxy, transition, judge
from .agent import Agent


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

    def __len__(self):
        return len(self.rewards)

    def __iter__(self):
        for board in self.boards:
            yield board


columns = [
    'boards_0',
    'boards_1',
    'coords',
    'rewards',
    'boards_2',
    'merits_2',
    'edges',
]


class Replay:

    def __init__(self, data: dict = dict()):
        self.data = data

    def __len__(self):
        return 0 if not self.data else len(self.data['rewards'])

    def __getitem__(self, key):
        return None if not self.data else self.data[key]

    def save(self, path: Path):
        assert self.data
        np.savez(path, **self.data)  # type: ignore

    @classmethod
    def load(cls, path: Path):
        memo = np.load(path)
        data = {col: memo[col] for col in columns}
        return cls(data)


def split(d: dict):
    d1, d2 = dict(), dict()
    for k, v in d.items():
        d1[k] = v[0::2]
        d2[k] = v[1::2]
    return d1, d2


def memoize(rollout: Rollout) -> tuple[Replay, Replay]:
    n = len(rollout)
    # 1. columnify
    p = {
        'boards_0': jnp.stack([onset] + rollout.boards[:-2]),
        'boards_1': jnp.stack(rollout.boards[:-1]),
        'coords': jnp.stack(rollout.coords),
        'rewards': jnp.stack(rollout.rewards)[:, None],
        'boards_2': jnp.stack(rollout.boards[1:]),
        'merits_2': jnp.stack([jnp.nan] * (n-2) + [0.] * 2)[:, None],
        'edges': jnp.stack([jnp.nan] * (n-1) + [0.] * 1)[:, None],
    }
    # 2. split
    p1, p2 = split(p)
    return Replay(p1), Replay(p2)


def collate(replays: Iterable[Replay]) -> Replay:
    memo = defaultdict(list)
    for col in columns:
        for replay in replays:
            if replay:
                memo[col].append(replay[col])
    # 3. concatentate
    data = {k: jnp.vstack(v) for k, v in memo.items()} if memo else dict()
    return Replay(data)


class Game:

    def __init__(self, agents: tuple[Agent, Agent]):
        self.agents = agents
        self.round  = 0
        self.winner = 9
        self.rollout = Rollout()

    def __len__(self):
        return len(self.rollout)

    @property
    def agent(self) -> Agent:
        return self.agents[self.round % 2]

    @property
    def board(self) -> Board:
        return self.rollout.last

    def evo(self, action: Action):
        if self.winner in (-1, 0, +1):
            raise ValueError('game over')
        stone, coord = action
        board = transition(self.board, stone, coord)
        winner = judge(board)
        reward = self.agent.eye(winner)
        self.rollout.append(coord, reward, board)
        self.winner = winner
        self.round += 1
        if winner in (-1, 0, +1):  # shadow tail of rollout
            self.rollout.append(proxy, self.agent.eye(winner), board)
        return winner


class Score:

    def __init__(self):
        self.wins = [0, 0, 0]
        self.lock = threading.Lock()

    @property
    def n(self) -> int:
        return sum(self.wins)

    def __call__(self, winner: Stone):
        with self.lock:
            self.wins[winner] += 1

    def __str__(self):
        return (
            f'X {self.wins[1]/self.n:.2%}\n'
            f'- {self.wins[0]/self.n:.2%}\n'
            f'O {self.wins[2]/self.n:.2%}\n'
            '--------'
        )


class Simulator:

    def __init__(self, n_processes: int = 1, n_threads: int = 1):
        self.n_processes = n_processes
        self.n_threads = n_threads

    def play(self, agents):
        game = Game(agents)
        while True:
            agent  = game.agent
            action = agent.act(game.board)
            winner = game.evo(action)
            if winner in (-1, 0, +1):
                break
        return winner, game.rollout

    def work(
        self,
        agents: tuple[Agent, Agent],
        stage: int,
        division: int,
        n_games: int,
    ) -> tuple[Replay, Replay]:
        key = jax.random.key(((stage + 3) * (division + 7) + 1) * 11 + 5)
        p1, p2 = agents[0].clone(), agents[1].clone()
        p1._key, p2._key = jax.random.split(key)
        replays_p1, replays_p2 = list(), list()
        for _ in range(n_games):
            _, rollout = self.play((p1, p2))
            replay_p1, replay_p2 = memoize(rollout)
            replays_p1.append(replay_p1)
            replays_p2.append(replay_p2)
        return collate(replays_p1), collate(replays_p2)

    def __call__(
        self,
        agents: tuple[Agent, Agent],
        stage: int,
        n_games: int,
        save: bool = False,
    ) -> tuple[Replay, Replay]:
        executor: Executor
        replays: Iterable[tuple[Replay, Replay]]
        if self.n_processes > 1 or self.n_threads > 1:
            if self.n_processes > 1:
                k = self.n_processes
                executor = ProcessPoolExecutor(max_workers=k)
            else:
                k = self.n_threads
                executor = ThreadPoolExecutor(max_workers=k)
            n = n_games // k
            m = n_games - (k - 1) * n
            with executor:
                replays = executor.map(
                    partial(self.work, agents),
                    repeat(stage), range(k), [n] * (k-1) + [m],
                )
            replays_p1, replays_p2 = list(zip(*replays))
            replay_p1 = collate(replays_p1)
            replay_p2 = collate(replays_p2)
        else:
            replay_p1, replay_p2 = self.work(agents, stage, 0, n_games)
        return replay_p1, replay_p2


class Loader:

    def __init__(
        self,
        replay: Replay,
        batch_size: int = 32,
        key: Array = jax.random.key(3),
    ):
        self.replay = replay
        self.batch_size = batch_size
        self._key = key

    @property
    def key(self):
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def permute(self, items):
        return jax.random.permutation(self.key, items)

    def __iter__(self):
        # 4. shuffle
        idx  = self.permute(jnp.arange(len(self.replay)))
        data = {k: v[idx] for k, v in self.replay.data.items()}
        i, b, n = 0, self.batch_size, len(self.replay)
        # 5. batch
        while i < n:
            yield {col: data[col][i:i+b] for col in columns}
            i += b
