import jax                                                        # type: ignore
from pathlib import Path
from g5.agent import Agent, ValueLearner, PolicyLearner
from g5.reward import Victory
from g5.value import MLPValue
from g5.policy import MLPPolicy
from g5.optim import CosineSchedule
from g5.game import Replay, Simulator, Loader
from util import Timer


n_stages = 100
n_games = 1000
n_epochs = 50
n_steps = n_stages * n_games * n_epochs * 100


folder = Path('data/dev/value/')
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


# folder = Path('data/dev/policy/')
# p1 = PolicyLearner(
#     stone=+1,
#     reward=Victory,
#     value=MLPValue(key=jax.random.key(21)),
#     policy=MLPPolicy(key=jax.random.key(87)),
#     epsilon=CosineSchedule(n_steps, 0.8, 0.01),
#     key=jax.random.key(1),
# )
# p2 = PolicyLearner(
#     stone=-1,
#     reward=Victory,
#     value=MLPValue(key=jax.random.key(43)),
#     policy=MLPPolicy(key=jax.random.key(65)),
#     epsilon=CosineSchedule(n_steps, 0.8, 0.01),
#     key=jax.random.key(2),
# )


try:
    task = f"Training {n_stages} stages x {n_games} games x {n_epochs} epochs"
    with Timer(task):
        for stage in range(1, n_stages+1):
            print(f"Stage {stage}")
            simulator = Simulator(agents=(p1, p2), folder=folder)
            replay_p1, replay_p2 = simulator(stage, n_games, True)
            for epoch in range(1, n_epochs+1):
                print(f"Epoch {epoch}")
                replay = Replay.load(folder / f'stage-{stage}_p1.npz')
                loader = Loader(replay, 512, jax.random.key(epoch))
                for data in loader: p1.obs(**data)
                replay = Replay.load(folder / f'stage-{stage}_p2.npz')
                loader = Loader(replay, 512, jax.random.key(epoch))
                for data in loader: p2.obs(**data)
            if stage % 10 == 0:
                p1.save(folder / f'p1.stage{stage}.msgpack')
                p2.save(folder / f'p2.stage{stage}.msgpack')
except Exception as exc:
    print("Training interrupted:")
    print(exc)
else:
    print("Training finished successfully.")
finally:
    p1.save(folder / 'p1.msgpack')
    p2.save(folder / 'p2.msgpack')
