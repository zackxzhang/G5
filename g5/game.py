import multiprocessing as mp
import numpy as np                                                # type: ignore
import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from jax import Array                                             # type: ignore
from collections import defaultdict
from collections.abc import Iterable
from itertools import repeat
from pathlib import Path
from .state import Stone, Board, Coord, Action, onset, proxy, transition, judge
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

    def __init__(self, data: dict | None = None):
        self.data = data if data else {col: jnp.array([]) for col in columns}

    def __len__(self):
        return len(self.data['rewards'])

    def __getitem__(self, key):
        return self.data[key]

    def save(self, path: Path):
        np.savez(path, **self.data)

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
            memo[col].append(replay[col])
    # 3. concatentate
    data = {k: jnp.vstack(v) for k, v in memo.items()}
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

    @property
    def n(self) -> int:
        return sum(self.wins)

    def __call__(self, winner: Stone):
        self.wins[winner] += 1

    def __str__(self):
        return (
            f'X {self.wins[1]/self.n:.2%}\n'
            f'- {self.wins[0]/self.n:.2%}\n'
            f'O {self.wins[2]/self.n:.2%}\n'
            '--------'
        )


class Simulator:

    def __init__(
        self,
        agents: tuple[Agent, Agent],
        folder: Path = Path('.'),
        device: str = 'cpu',
        n_procs: int = 8,
    ):
        self.score  = Score()
        self.agents = agents
        self.folder = folder
        self.device = device
        self.n_procs = n_procs

    def play(self):
        game = Game(self.agents)
        while True:
            agent  = game.agent
            action = agent.act(game.board)
            winner = game.evo(action)
            if winner in (-1, 0, +1):
                self.score(winner)
                print(f"The game took {len(game)} steps.")
                break
        return game.rollout

    def work(
        self,
        stage: int,
        division: int,
        n_games: int,
    ) -> tuple[Replay, Replay]:
        key = jax.random.key(((stage + 3) * (division + 7) + 1) * 11 + 5)
        key0, key1 = jax.random.split(key)
        self.agents[0]._key = key0
        self.agents[1]._key = key1
        replays_p1, replays_p2 = list(), list()
        for _ in range(n_games):
            replay_p1, replay_p2 = memoize(self.play())
            replays_p1.append(replay_p1)
            replays_p2.append(replay_p2)
        print(self.score)
        return collate(replays_p1), collate(replays_p2)

    def __call__(
        self,
        stage: int,
        n_games: int,
        save: bool = False,
    ) -> tuple[Replay, Replay]:
        match self.device:
            case 'cpu':
                k = self.n_procs
                n = n_games // k
                m = n_games - (k - 1) * n
                with mp.Pool(k) as pool:
                    replays: list[tuple[Replay, Replay]] = pool.starmap(
                        self.work,
                        zip(repeat(stage), range(k), [n] * (k-1) + [m]),
                    )
                replays_p1, replays_p2 = list(zip(*replays))
                replay_p1 = collate(replays_p1)
                replay_p2 = collate(replays_p2)
            case _:
                replay_p1, replay_p2 = self.work(stage, 0, n_games)
        if save:
            replay_p1.save(self.folder / f'stage-{stage}_p1.npz')
            replay_p2.save(self.folder / f'stage-{stage}_p2.npz')
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
