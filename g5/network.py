import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
import jax.scipy as jsp                                           # type: ignore
from jax.scipy.special import logsumexp                           # type: ignore


def relu(x):
  return jnp.maximum(0, x)


def mlp_init_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = jax.random.split(key)
    return (
        scale * jax.random.normal(w_key, (n, m)),
        scale * jax.random.normal(b_key, (n,))
    )


def mlp_init_network_params(sizes, key):
    keys = jax.random.split(key, len(sizes))
    return [
        mlp_init_layer_params(m, n, k)
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]
