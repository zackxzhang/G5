import jax                                                        # type: ignore
import time
from pathlib import Path
from g5.agent import Learner, ValueLearner, PolicyLearner
from g5.reward import Victory
from g5.value import MLPValue
from g5.policy import MLPPolicy
from g5.optim import CosineSchedule
from g5.game import Replay, Simulator, Loader


class Timer:

    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *args):
        self.end = time.perf_counter()
        print('-' * 64)
        print(self.message + ' takes time: ' + str(self.end - self.start))
        print('-' * 64)


if __name__ == '__main__':

    n_stages = 20
    n_games = 200
    n_steps = n_stages * n_games * 150

    p1: Learner
    p2: Learner

    folder = Path('data/value/')
    # p1 = ValueLearner.load(folder / 'p1.msgpack')
    # p2 = ValueLearner.load(folder / 'p2.msgpack')
    p1 = ValueLearner(
        stone=+1,
        reward=Victory,
        value=MLPValue(key=jax.random.key(21)),
        epsilon=CosineSchedule(n_steps, 0.8, 0.01),
        key=jax.random.key(1),
    )
    p2 = ValueLearner(
        stone=-1,
        reward=Victory,
        value=MLPValue(key=jax.random.key(43)),
        epsilon=CosineSchedule(n_steps, 0.8, 0.01),
        key=jax.random.key(2),
    )

    # folder = Path('data/policy/')
    # p1 = PolicyLearner.load(folder / 'p1.msgpack')
    # p2 = PolicyLearner.load(folder / 'p2.msgpack')
    # p1 = PolicyLearner(
    #     stone=+1,
    #     reward=Victory,
    #     value=MLPValue(key=jax.random.key(21)),
    #     policy=MLPPolicy(key=jax.random.key(87)),
    #     key=jax.random.key(1),
    # )
    # p2 = PolicyLearner(
    #     stone=-1,
    #     reward=Victory,
    #     value=MLPValue(key=jax.random.key(43)),
    #     policy=MLPPolicy(key=jax.random.key(65)),
    #     key=jax.random.key(2),
    # )

    with Timer(f"Training for {n_stages} stages ({n_games} games each)"):

        for stage in range(0, n_stages):

            simulator = Simulator(agents=(p1, p2), folder=folder)
            replay_p1, replay_p2 = simulator(
                stage=stage,
                n_games=n_games,
                save=True,
            )

            replay = Replay.load(folder / f'stage-{stage}_p1.npz')
            loader = Loader(replay, batch_size=100)
            for data in loader:
                print([arr.shape for arr in data.values()])
                p1.obs(**data)

            replay = Replay.load(folder / f'stage-{stage}_p2.npz')
            loader = Loader(replay, batch_size=100)
            for data in loader:
                print([arr.shape for arr in data.values()])
                p2.obs(**data)

    p1.save(folder / 'p1.msgpack')
    p2.save(folder / 'p2.msgpack')
