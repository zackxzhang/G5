import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from abc import ABC, abstractmethod
from .state import Board
from .network import relu, logsumexp, mlp_init_network_params


class Value(ABC):

    @abstractmethod
    def predict(self, boards: list[Board]):
        pass

    def __call__(self, boards: list[Board]):
        return self.predict(boards)

    @abstractmethod
    def update(self, boards_0: list[Board], rewards, boards_1: list[Board]):
        pass


def mlp_predict(params, board):
    acts = board
    for w, b in params[:-1]:
        outs = jnp.dot(w, acts) + b
        acts = relu(outs)
    w, b = params[-1]
    return jnp.dot(w, acts) + b


mlp_predict_batch = jax.vmap(mlp_predict, in_axes=(None, 0))


def mlp_loss(params, boards_0, rewards, boards_1, gamma=1.0):
    values_0 = mlp_predict_batch(params, boards_0)
    values_1 = jax.lax.stop_gradient(mlp_predict_batch(params, boards_1))
    advantages = rewards + gamma * values_1 - values_0
    return advantages ** 2


@jax.jit
def mlp_step(params, boards_0, rewards, boards_1, alpha=1e-2):
    grads = jax.grad(loss)(params, boards_0, rewards, boards_1)
    return [
        (w - alpha * dw, b - alpha * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


class MLPValue(Value):

    def __init__(self,
        layer_sizes=[225, 900, 900, 225, 1],
        key=jax.random.key(ord('v')),
    ):
        self.params = mlp_init_network_params(layer_sizes, key)

    def predict(self, boards: list[Board]):
        return mlp_predict_batch(self.params, boards)

    def update(self, boards_0: list[Board], rewards, boards_1: list[Board]):
        self.params = mlp_step(self.params, boards_0, rewards, boards_1)


class CNNValue(Value):

    ...
