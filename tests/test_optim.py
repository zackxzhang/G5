import matplotlib.pyplot as plt                                   # type: ignore
import unittest
from g5.optim import LinearSchedule, CosineSchedule, ExponentialSchedule


class OptimTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        plt.style.use('ggplot')

    def test_linear_schedule(self):
        s = LinearSchedule(8_000, 1e-2, 1e-4)
        plt.plot([s() for _ in range(10_000)], label='linear')

    def test_cosine_schedule(self):
        s = CosineSchedule(8_000, 1e-2, 1e-4)
        plt.plot([s() for _ in range(10_000)], label='cosine')

    def test_exponential_schedule(self):
        s = ExponentialSchedule(8_000, 1e-2, 1e-4, 0.9995)
        plt.plot([s() for _ in range(10_000)], label='exponential')

    @classmethod
    def tearDownClass(cls):
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('learning rate')
        plt.title('Learning Rate Schedules')
        plt.savefig('schedules.png', dpi=300)
