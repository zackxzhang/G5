import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from abc import ABC, abstractmethod
from .state import Board, Boards
from .network import relu, logsumexp, mlp_init_network_params


class Value(ABC):

    @abstractmethod
    def predict(self, board: Board):
        pass

    @abstractmethod
    def update(self, boards_0: Boards, rewards, boards_1: Boards):
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
    return rewards + gamma * values_1 - values_0


@jax.jit
def mlp_step(params, boards_0, rewards, boards_1, alpha=1e-2):
    grads = jax.grad(loss)(params, boards_0, boards_1, rewards)
    return [
        (w - alpha * dw, b - alpha * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


class MLPValue(Value):

    def __init__(self,
        layer_sizes=[225, 900, 900, 225, 1],
        key=jax.random.key(0)
    ):
        self.params = mlp_init_network_params(layer_sizes, key)

    def predict(self, board: Board):
        return mlp_predict(self.params, board)

    def predict_batch(self, boards: Boards):
        return mlp_predict_batch(self.params, boards)

    def update(self, boards_0: Boards, rewards, boards_1: Boards):
        self.params = mlp_step(self.params, boards_0, rewards, boards_1)


class CNNValue(Value):

    ...
