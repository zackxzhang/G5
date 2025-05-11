from pathlib import Path
from g5.agent import Agent, ValueLearner, PolicyLearner
from g5.reward import Victory
from g5.value import MLPValue
from g5.policy import MLPPolicy
from g5.optim import CosineSchedule
from g5.game import Replay, Simulator, Loader


p1: Agent
p2: Agent

n_stages = 10
n_games = 100
n_steps = n_stages * n_games * 100


if __name__ == '__main__':

    folder = Path('data/value/')
    p1 = ValueLearner(
        stone=+1,
        reward=Victory,
        value=MLPValue(seed=21),
        epsilon=CosineSchedule(n_steps, 0.8, 0.01),
        seed=1,
    )
    p2 = ValueLearner(
        stone=-1,
        reward=Victory,
        value=MLPValue(seed=43),
        epsilon=CosineSchedule(n_steps, 0.8, 0.01),
        seed=2,
    )

    # folder = Path('data/policy/')
    # p1 = PolicyLearner(
    #     stone=+1,
    #     reward=Victory,
    #     value=MLPValue(seed=21),
    #     policy=MLPPolicy(seed=87),
    #     seed=1,
    # )
    # p2 = PolicyLearner(
    #     stone=-1,
    #     reward=Victory,
    #     value=MLPValue(seed=43),
    #     policy=MLPPolicy(seed=65),
    #     seed=2,
    # )

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
