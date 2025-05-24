import jax                                                        # type: ignore
import unittest
from pathlib import Path
from g5.reward import Victory
from g5.value import MLPValue
from g5.policy import MLPPolicy
from g5.agent import ValueLearner, PolicyLearner
from g5.optim import CosineSchedule


class AgentTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.folder = Path('data/test/')
        cls.data = (
            jax.random.randint(jax.random.key(0), (64, 15, 15), -1, +2),
            jax.random.randint(jax.random.key(0), (64, 15, 15), -1, +2),
            jax.random.randint(jax.random.key(0), (64, 2), 0, 15),
            jax.random.normal(jax.random.key(0), (64, 1)),
            jax.random.randint(jax.random.key(0), (64, 15, 15), -1, +2),
            jax.random.normal(jax.random.key(0), (64, 1)),
            jax.random.normal(jax.random.key(0), (64, 1)),
        )

    @classmethod
    def make_value(cls):
        p1 = ValueLearner(
            stone=+1,
            reward=Victory,
            value=MLPValue(key=jax.random.key(21)),
            epsilon=CosineSchedule(1000, 0.8, 0.01),
            key=jax.random.key(1),
        )
        p2 = ValueLearner(
            stone=-1,
            reward=Victory,
            value=MLPValue(key=jax.random.key(43)),
            epsilon=CosineSchedule(1000, 0.8, 0.01),
            key=jax.random.key(2),
        )
        return p1, p2

    @classmethod
    def make_policy(cls):
        p1 = PolicyLearner(
            stone=+1,
            reward=Victory,
            value=MLPValue(key=jax.random.key(21)),
            policy=MLPPolicy(key=jax.random.key(87)),
            epsilon=CosineSchedule(1000, 0.8, 0.01),
            key=jax.random.key(1),
        )
        p2 = PolicyLearner(
            stone=-1,
            reward=Victory,
            value=MLPValue(key=jax.random.key(43)),
            policy=MLPPolicy(key=jax.random.key(65)),
            epsilon=CosineSchedule(1000, 0.8, 0.01),
            key=jax.random.key(2),
        )
        return p1, p2

    def test_value_codec(self):
        p1, p2 = self.make_value()
        p1.save(self.folder / 'p1.msgpack')
        p2.save(self.folder / 'p2.msgpack')
        p3 = ValueLearner.load(self.folder / 'p1.msgpack')
        p4 = ValueLearner.load(self.folder / 'p2.msgpack')
        self.assertTrue((p1.value.params[0][0] == p3.value.params[0][0]).all())
        self.assertTrue((p2.value.params[0][0] == p4.value.params[0][0]).all())

    def test_policy_codec(self):
        p1, p2 = self.make_policy()
        p1.save(self.folder / 'p1.msgpack')
        p2.save(self.folder / 'p2.msgpack')
        p3 = PolicyLearner.load(self.folder / 'p1.msgpack')
        p4 = PolicyLearner.load(self.folder / 'p2.msgpack')
        self.assertTrue((p1.value.params[0][0] == p3.value.params[0][0]).all())
        self.assertTrue((p2.value.params[0][0] == p4.value.params[0][0]).all())

    def test_value_eval(self):
        p1, p2 = self.make_value()
        params = p1.value.params
        p1.obs(*self.data)
        self.assertFalse((params[0][0] == p1.value.params[0][0]).all())
        p1 = p1.eval()
        params = p1.value.params
        p1.obs(*self.data)
        self.assertTrue((params[0][0] == p1.value.params[0][0]).all())

    def test_policy_eval(self):
        p1, p2 = self.make_policy()
        params_v = p1.value.params
        params_p = p1.policy.params
        p1.obs(*self.data)
        self.assertFalse((params_v[0][0] == p1.value.params[0][0]).all())
        self.assertFalse((params_p[0][0] == p1.policy.params[0][0]).all())
        p1 = p1.eval()
        params_v = p1.value.params
        params_p = p1.policy.params
        p1.obs(*self.data)
        self.assertTrue((params_v[0][0] == p1.value.params[0][0]).all())
        self.assertTrue((params_p[0][0] == p1.policy.params[0][0]).all())
