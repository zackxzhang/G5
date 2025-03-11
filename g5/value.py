import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from abc import ABC, abstractmethod
from .state import Board
from .network import relu, logsumexp, mlp_init_network_params


class Value(ABC):

    @abstractmethod
    def predicts(self, board):
        pass

    @abstractmethod
    def predict(self, boards):
        pass

    def __call__(self, boards):
        return self.predict(boards)

    @abstractmethod
    def update(self, boards_0, rewards, boards_1):
        pass


def mlp_predict(params, board):
    acts = board.ravel()
    for w, b in params[:-1]:
        outs = jnp.dot(w, acts) + b
        acts = relu(outs)
    w, b = params[-1]
    return jnp.dot(w, acts) + b


mlp_predict_batch = jax.vmap(mlp_predict, in_axes=(None, 0))


def advantage(values_0, rewards, values_2, gamma=1.0):
    return rewards + gamma * values_2 - values_0


def mlp_loss(params, boards_0, rewards, boards_2, merits_2):
    values_0 = mlp_predict_batch(params, boards_0)
    values_2 = jnp.where(
        merits_2.isnan(),
        mlp_predict_batch(params, boards_2),
        merits_2,
    )
    advantages = advantage(values_0, rewards, jax.lax.stop_gradient(values_2))
    return jax.sum(advantages**2)


@jax.jit
def mlp_step(params, boards_0, rewards, boards_2, merits_2, alpha=1e-2):
    grads = jax.grad(mlp_loss)(params, boards_0, rewards, boards_2, merits_2)
    return [
        (w - alpha * dw, b - alpha * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


class MLPValue(Value):

    def __init__(
        self,
        layer_sizes=[225, 900, 900, 225, 1],
        seed=5,
    ):
        self.params = mlp_init_network_params(layer_sizes, jax.random.key(seed))

    def predicts(self, board):
        return mlp_predict(self.params, board)

    def predict(self, boards):
        return mlp_predict_batch(self.params, boards)

    def update(self, boards_0, rewards, boards_2, merits_2):
        self.params = mlp_step(
            self.params,
            boards_0,
            rewards,
            boards_2,
            values_2
        )


class CNNValue(Value):

    ...


class G5Value(Value):

    ...
