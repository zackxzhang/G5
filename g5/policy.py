import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from jax import Array                                             # type: ignore
from abc import ABC, abstractmethod
from .state import Board, Coord
from .network import PyTree, relu, logsumexp, mlp_init_network_params
from .value import advantage


class Policy(ABC):

    params: PyTree
    _key: Array

    @abstractmethod
    def predicts(self, board) -> Array:
        pass

    @abstractmethod
    def predict(self, boards) -> Array:
        pass

    def __call__(self, boards):
        return self.predict(boards)

    @abstractmethod
    def update(self, boards, coords, advantages):
        pass

    @property
    def key(self):
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def encode(self) -> PyTree:
        return {
            'class': self.__class__.__name__,
            'params': self.params,
        }

    @classmethod
    def decode(cls, data: PyTree):
        match data['class']:
            case 'MLPPolicy':
                return MLPPolicy(data['params'])
            # case 'UNetPolicy':
            #     return UNetPolicy(data['params'])
            # case 'G5Policy':
            #     return G5Policy(data['params'])
            case _:
                raise ValueError(f"no policy class named {data['class']}")


def mlp_predict(params, board):
    acts = board.ravel()
    for w, b in params[:-1]:
        outs = jnp.dot(w, acts) + b
        acts = relu(outs)
    w, b = params[-1]
    logits = jnp.dot(w, acts) + b
    logpbs = logits - logsumexp(logits)
    return logpbs.reshape((15, 15))


mlp_predict_batch = jax.vmap(mlp_predict, in_axes=(None, 0))


def critic(value_fn, boards_0, rewards, boards_2, merits_2, edges):
    values_0 = value_fn(boards_0)
    values_2 = jnp.where(
        jnp.isnan(merits_2),
        value_fn(boards_2),
        merits_2,
    )
    advantages = jnp.where(
        jnp.isnan(edges),
        advantage(values_0, rewards, values_2),
        edges,
    )
    return advantages


def mlp_loss(params, boards, coords, advantages):
    n = len(coords)
    logpbs = mlp_predict_batch(params, boards)
    logpas = logpbs[jnp.arange(n), coords[:, 0], coords[:, 1]][:, None]
    return jnp.sum(advantages * logpas)


@jax.jit
def mlp_step(params, boards, coords, advantages, beta=1e-2):
    grads = jax.grad(mlp_loss)(params, boards, coords, advantages)
    return [
        (w - beta * dw, b - beta * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


class MLPPolicy(Policy):

    def __init__(self, params: PyTree | None = None, key=jax.random.key(6)):
        self._key = key
        self.params = (
            params if params else
            mlp_init_network_params([225, 900, 900, 225], self.key)
        )

    def predicts(self, board):
        return mlp_predict(self.params, board)

    def predict(self, boards):
        return mlp_predict_batch(self.params, boards)

    def update(self, boards, coords, advantages):
        self.params = mlp_step(self.params, boards, coords, advantages)


class UNetPolicy(Policy):

    ...


class G5Policy(Policy):

    ...
