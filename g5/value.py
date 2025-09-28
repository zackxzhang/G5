import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from abc import ABC, abstractmethod
from functools import partial
from .hint import Board, Array, Layers, PyTree, Key
from .network import (
    step, move, tanh, flatten,
    encode_layers, decode_layers,
    InputLayer, Conv2DLayer, MaxPoolLayer, FlattenLayer, DenseLayer,
    mlp_init_network_params, mlp_forward, mlp_forward_batch,
    cnn_init_network_params, cnn_forward, cnn_forward_batch,
)
from .codec import encode_key, decode_key


class Value(ABC):

    params: PyTree
    _key: Key

    def __init__(self):
        self.learnable = True

    def eval(self):
        self.learnable = False
        return self

    @abstractmethod
    def predicts(self, board) -> Array:
        pass

    @abstractmethod
    def predict(self, boards) -> Array:
        pass

    @abstractmethod
    def __call__(self, boards) -> Array:
        pass

    @abstractmethod
    def update(self, boards_0, rewards, boards_1, merits_2):
        pass

    @property
    def key(self):
        self._key, subkey = jax.random.split(self._key)
        return subkey

    @abstractmethod
    def encode(self) -> PyTree:
        pass

    @classmethod
    def decode(cls, data: PyTree):
        genre = data.pop('class')
        match genre:
            case 'MLPValue':
                return MLPValue.decode(data)
            case 'CNNValue':
                return CNNValue.decode(data)
            # case 'G5Value':
            #     return G5Value(data)
            case _:
                raise ValueError(f"unknown value class: {genre}")


@jax.jit
def mlp_predict(params, board):
    return tanh(mlp_forward(params, board.ravel()))


@jax.jit
def mlp_predict_batch(params, boards):
    return tanh(mlp_forward_batch(params, flatten(boards)))


def advantage(values_0, rewards, values_2, gamma: float = 1.0):
    return rewards + gamma * values_2 - values_0


def mlp_loss(params, params_p, boards_0, rewards, boards_2, merits_2):
    values_0 = mlp_predict_batch(params, boards_0)
    values_2 = jnp.where(
        jnp.isnan(merits_2),
        mlp_predict_batch(params_p, boards_2),
        merits_2,
    )
    advantages = advantage(values_0, rewards, jax.lax.stop_gradient(values_2))
    return jnp.mean(advantages**2)


@jax.jit
def mlp_step(params, params_p, boards_0, rewards, boards_2, merits_2, alpha):
    grads = jax.grad(mlp_loss)(
        params, params_p, boards_0, rewards, boards_2, merits_2
    )
    return [
        [W - alpha * dW, b - alpha * db]
        for (W, b), (dW, db) in zip(params, grads)
    ]


mlp_default_sizes = [225, 900, 3600, 900, 225, 1]
# mlp_default_sizes = [225, 1800, 3600, 1800, 900, 450, 225, 1]


class MLPValue(Value):

    def __init__(
        self,
        params: PyTree | None = None,
        key: Key = jax.random.key(5),
    ):
        super().__init__()
        self._key = key
        self.params = (
            params if params else
            mlp_init_network_params(mlp_default_sizes, self.key)
        )
        self.params_p = self.params

    def encode(self) -> PyTree:
        return {
            'class':  self.__class__.__name__,
            'params': self.params,
            'key':    encode_key(self._key),
        }

    @classmethod
    def decode(cls, data: PyTree):
        params = data['params']
        key = decode_key(data['key'])
        return cls(params, key)

    def predicts(self, board):
        return mlp_predict(self.params, board)

    def predict(self, boards):
        return mlp_predict_batch(self.params, boards)

    def __call__(self, boards):
        return mlp_predict_batch(self.params_p, boards)

    def update(self, boards_0, rewards, boards_2, merits_2):
        if self.learnable:
            self.params = mlp_step(
                self.params,
                self.params_p,
                boards_0,
                rewards,
                boards_2,
                merits_2,
                alpha=1e-3,
            )
            self.params_p = move(self.params_p, self.params, beta=0.999)
        else:
            pass


@partial(jax.jit, static_argnames=('layers',))
def cnn_predict(params, layers, board):
    intake = jnp.expand_dims(board.astype(float), axis=-3)
    return cnn_forward(params, layers, intake)


@partial(jax.jit, static_argnames=('layers',))
def cnn_predict_batch(params, layers, boards):
    inputs = jnp.expand_dims(boards.astype(float), axis=-3)
    return cnn_forward_batch(params, layers, inputs)


@partial(jax.jit, static_argnames=('layers',))
def cnn_loss(params, params_p, layers, boards_0, rewards, boards_2, merits_2):
    values_0 = cnn_predict_batch(params, layers, boards_0)
    values_2 = jnp.where(
        jnp.isnan(merits_2),
        cnn_predict_batch(params_p, layers, boards_2),
        merits_2,
    )
    advantages = advantage(values_0, rewards, jax.lax.stop_gradient(values_2))
    return jnp.sum(advantages**2)


@partial(jax.jit, static_argnames=('layers',))
def cnn_step(
    params, params_p, layers, boards_0, rewards, boards_2, merits_2, alpha,
):
    grads = jax.grad(cnn_loss)(
        params,
        params_p,
        layers,
        boards_0,
        rewards,
        boards_2,
        merits_2,
    )
    return step(params, grads, alpha)


cnn_default_layers: Layers = (
    InputLayer(                                    # type: ignore[call-overload]
        shape=(None, 1, 15, 15),
    ),
    Conv2DLayer(                                   # type: ignore[call-overload]
        channels=512,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='SAME',
        activation='relu',
    ),
    MaxPoolLayer(                                  # type: ignore[call-overload]
        pool_size=(3, 3),
        strides=(3, 3),
        padding='VALID',
    ),
    Conv2DLayer(                                   # type: ignore[call-overload]
        channels=1024,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='VALID',
        activation='relu',
    ),
    FlattenLayer(),                                # type: ignore[call-overload]
    DenseLayer(                                    # type: ignore[call-overload]
        units=2048,
        activation='relu',
    ),
    DenseLayer(                                    # type: ignore[call-overload]
        units=512,
        activation='relu',
    ),
    DenseLayer(                                    # type: ignore[call-overload]
        units=32,
        activation='relu',
    ),
    DenseLayer(                                    # type: ignore[call-overload]
        units=1,
        activation='sigmoid',
    ),
)


class CNNValue(Value):

    def __init__(self,
        layers: Layers | None = None,
        params: PyTree | None = None,
        key: Key = jax.random.key(7),
    ):
        super().__init__()
        self._key = key
        self.layers = layers if layers else cnn_default_layers
        self.params = (
            params if params else
            cnn_init_network_params(self.layers, self.key)
        )
        self.params_p = self.params

    def encode(self):
        return {
            'class':  self.__class__.__name__,
            'layers': encode_layers(self.layers),
            'params': self.params,
            'key':    encode_key(self._key),
        }

    @classmethod
    def decode(cls, data: PyTree):
        layers = decode_layers(data['layers'])
        params = data['params']
        key = decode_key(data['key'])
        return cls(layers, params, key)

    def predicts(self, board):
        return cnn_predict(self.params, self.layers, board)

    def predict(self, boards):
        return cnn_predict_batch(self.params, self.layers, boards)

    def __call__(self, boards):
        return cnn_predict_batch(self.params_p, self.layers, boards)

    def update(self, boards_0, rewards, boards_2, merits_2):
        if self.learnable:
            self.params = cnn_step(
                self.params,
                self.params_p,
                self.layers,
                boards_0,
                rewards,
                boards_2,
                merits_2,
                alpha=1e-3,
            )
            self.params_p = move(self.params_p, self.params, beta=0.999)
        else:
            pass


class G5Value(Value):

    ...
