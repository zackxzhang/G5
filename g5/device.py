import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore


cpus = jax.devices('cpu')
gpus = jax.devices('gpu')


try:
    backend = 'gpu'
    device = gpus[0]
    with jax.default_device(device):
        a = jnp.ones((3, 3))
        b = a @ a
except Exception as exc:
    backend = 'cpu'
    device = cpus[0]


print(f'backend: {backend}')
print(f'device: {device}')
