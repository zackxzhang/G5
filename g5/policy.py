import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
from jax import Array                                             # type: ignore
from abc import ABC, abstractmethod
from .hint import Board, Coord, PyTree, Key
from .network import logsumexp, mlp_init_network_params, mlp_forward
from .value import advantage
from .codec import encode_key, decode_key


class Policy(ABC):

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

    def __call__(self, boards):
        return self.predict(boards)

    @abstractmethod
    def update(self, boards, coords, advantages):
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
            case 'MLPPolicy':
                return MLPPolicy.decode(data)
            # case 'UNetPolicy':
            #     return UNetPolicy.decode(data)
            # case 'G5Policy':
            #     return G5Policy.decode(data)
            case _:
                raise ValueError(f"no policy class: {genre}")


@jax.jit
def mlp_predict(params, board):
    logits = mlp_forward(params, board.ravel())
    logpbs = logits - logsumexp(logits)
    return logpbs.reshape((15, 15))


mlp_predict_batch = jax.jit(jax.vmap(mlp_predict, in_axes=(None, 0)))


@jax.jit
def critic(values_0, rewards, values_2, merits_2, edges):
    values_2 = jnp.where(
        jnp.isnan(merits_2),
        values_2,
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
    return jnp.mean(advantages * logpas)


@jax.jit
def mlp_step(params, boards, coords, advantages, alpha=1e-2):
    grads = jax.grad(mlp_loss)(params, boards, coords, advantages)
    return [
        [W - alpha * dW, b - alpha * db]
        for (W, b), (dW, db) in zip(params, grads)
    ]


mlp_default_sizes = [225, 900, 3600, 900, 225]
# mlp_default_sizes = [225, 1800, 3600, 1800, 900, 450, 225]


class MLPPolicy(Policy):

    def __init__(
        self,
        params: PyTree | None = None,
        key: Key = jax.random.key(6),
    ):
        super().__init__()
        self._key = key
        self.params = (
            params if params else
            mlp_init_network_params(mlp_default_sizes, self.key)
        )

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

    def update(self, boards, coords, advantages):
        if self.learnable:
            self.params = mlp_step(self.params, boards, coords, advantages)
        else:
            pass


class UNetPolicy(Policy):

    ...


class G5Policy(Policy):

    ...
