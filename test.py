import jax                                                        # type: ignore
from pathlib import Path
from g5.agent import Amateur, ValueLearner, PolicyLearner
from g5.reward import Victory
from g5.game import Simulator


folder = Path('data/value/')
p1 = ValueLearner.load(folder / 'p1.msgpack').eval()
p2 = ValueLearner.load(folder / 'p2.msgpack').eval()
# folder = Path('data/policy/')
# p1 = PolicyLearner.load(folder / 'p1.msgpack').eval()
# p2 = PolicyLearner.load(folder / 'p2.msgpack').eval()
p3 = Amateur(+1, Victory, key=jax.random.key(3))
p4 = Amateur(-1, Victory, key=jax.random.key(4))
matches = ((p3, p4), (p1, p4), (p3, p2), (p1, p2))


for agents in matches:
    simulator = Simulator(agents=agents)
    for _ in range(1_000):
        simulator.play()
    print(simulator.score)
