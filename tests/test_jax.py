import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
import time
import unittest


def run_on_backend(backend):
    with jax.default_device(jax.devices(backend)[0]):
        for size in [10, 100, 1000, 5000]:
            x = jnp.ones((size, size))
            for _ in range(5):
                result = jnp.matmul(x, x)
                result.block_until_ready()
            start = time.time()
            for _ in range(10):
                result = jnp.matmul(x, x)
                result.block_until_ready()
            elapsed = time.time() - start
            print(
                f"on {backend}, {size:>5} x {size:>5}: "
                f"{elapsed/10:.6f} seconds per iteration"
            )


class JaxTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("JAX version:", jax.__version__)
        print("JAX backend:", jax.default_backend())
        print("JAX devices:", jax.devices())

    def test_device(self):
        run_on_backend('cpu')
        try:
            run_on_backend('gpu')
        except Exception as exc:
            print('No viable gpu devices')
            print(exc)

    def test_matmul(self):
        a = jnp.ones((7, 7))
        print(a @ a)
        b = jnp.ones((8, 8))
        print(b @ b)

    def test_grad(self):
        params = jnp.array([1., 2.])
        def loss(params, data):
            return jnp.sum(3*params[0]) + jnp.sum(data[1])
        self.assertTrue(
            (jax.grad(loss)(params, 4*params) == jnp.array([3, 0])).all()
        )
