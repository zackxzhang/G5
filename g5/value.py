import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from jax import Array                                             # type: ignore
from abc import ABC, abstractmethod
from .state import Board
from .network import PyTree, relu, mlp_init_network_params


class Value(ABC):

    params: PyTree
    _key: Array

    def __init__(self):
        self.learnable = True

    def eval(self):
        self.learnable = False

    @abstractmethod
    def predicts(self, board) -> Array:
        pass

    @abstractmethod
    def predict(self, boards) -> Array:
        pass

    def __call__(self, boards):
        return self.predict(boards)

    @abstractmethod
    def update(self, boards_0, rewards, boards_1, merits_2):
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
            case 'MLPValue':
                return MLPValue(data['params'])
            # case 'CNNValue':
            #     return CNNValue(data['params'])
            # case 'G5Value':
            #     return G5Value(data['params'])
            case _:
                raise ValueError(f"no value class named {data['class']}")


@jax.jit
def mlp_predict(params, board):
    acts = board.ravel()
    for w, b in params[:-1]:
        outs = jnp.dot(w, acts) + b
        acts = relu(outs)
    w, b = params[-1]
    return jnp.dot(w, acts) + b


mlp_predict_batch = jax.jit(jax.vmap(mlp_predict, in_axes=(None, 0)))


def advantage(values_0, rewards, values_2, gamma=1.0):
    return rewards + gamma * values_2 - values_0


def mlp_loss(params, boards_0, rewards, boards_2, merits_2):
    values_0 = mlp_predict_batch(params, boards_0)
    values_2 = jnp.where(
        jnp.isnan(merits_2),
        mlp_predict_batch(params, boards_2),
        merits_2,
    )
    advantages = advantage(values_0, rewards, jax.lax.stop_gradient(values_2))
    return jnp.sum(advantages**2)


@jax.jit
def mlp_step(params, boards_0, rewards, boards_2, merits_2, alpha=1e-2):
    grads = jax.grad(mlp_loss)(params, boards_0, rewards, boards_2, merits_2)
    return [
        [w - alpha * dw, b - alpha * db]
        for (w, b), (dw, db) in zip(params, grads)
    ]


class MLPValue(Value):

    def __init__(self, params: PyTree | None = None, key=jax.random.key(5)):
        super().__init__()
        self._key = key
        self.params = (
            params if params else
            mlp_init_network_params([225, 900, 3600, 900, 225, 1], self.key)
        )

    def predicts(self, board):
        return mlp_predict(self.params, board)

    def predict(self, boards):
        return mlp_predict_batch(self.params, boards)

    def update(self, boards_0, rewards, boards_2, merits_2):
        if self.learnable:
            self.params = mlp_step(
                self.params,
                boards_0,
                rewards,
                boards_2,
                merits_2,
            )
        else:
            pass


class CNNValue(Value):

    ...


class G5Value(Value):

    ...
