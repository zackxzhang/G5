import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from abc import ABC, abstractmethod
from .state import Board, Boards
from .network import relu, logsumexp, mlp_init_network_params


class Policy(ABC):

    @abstractmethod
    def predict(self, board: Board):
        pass

    @abstractmethod
    def update(self, boards: Boards, advantages):
        pass


def mlp_predict(params, board):
    acts = board
    for w, b in params[:-1]:
        outs = jnp.dot(w, acts) + b
        acts = relu(outs)
    w, b = params[-1]
    logits = jnp.dot(w, acts) + b
    logpbs = logits - logsumexp(logits)
    return logpbs


mlp_predict_batch = jax.vmap(mlp_predict, in_axes=(None, 0))


def mlp_loss(params, boards, actions, advantages, gamma=1.0):
    logpbs = mlp_predict_batch(params, boards)
    logpas = logpbs[actions]
    return jax.lax.stop_gradient(advantages) * logpas


@jax.jit
def mlp_step(params, boards, actions, advantages, beta=1e-2):
    grads = jax.grad(loss)(params, boards, advantages)
    return [
        (w - beta * dw, b - beta * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


class MLPPolicy(Policy):

    def __init__(self,
        layer_sizes=[225, 900, 900, 225],
        key=jax.random.key(ord('p')),
    ):
        self.params = mlp_init_network_params(layer_sizes, key)

    def predict(self, board: Board):
        return mlp_predict(self.params, board)

    def predict_batch(self, boards: Boards):
        return mlp_predict_batch(self.params, boards)

    def update(self, boards: Boards, advantages):
        self.params = mlp_step(self.params, boards, advantages)


class UNetPolicy(Policy):

    ...
