import json
import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
import numpy as np                                                # type: ignore
from jax.scipy.special import logsumexp                           # type: ignore
from jax.nn import relu, sigmoid                                  # type: ignore
from functools import partial
from typing import NamedTuple, Any, TypeAlias


PyTree: TypeAlias = Any


def mlp_init_layer_params(m, n, key, scale=1e-2):
    W_key, b_key = jax.random.split(key)
    return [
        scale * jax.random.normal(W_key, (n, m)),
        scale * jax.random.normal(b_key, (n,  )),
    ]


def mlp_init_network_params(sizes, key):
    keys = jax.random.split(key, len(sizes))
    return [
        mlp_init_layer_params(m, n, k)
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


@jax.jit
def mlp_forward(params, x):
    for W, b in params[:-1]:
        z = W @ x + b
        x = relu(z)
    W, b = params[-1]
    return W @ x + b


mlp_forward_batch = jax.jit(jax.vmap(mlp_forward, in_axes=(None, 0)))


def conv2d_output_shape(lhs, rhs, strides, padding):
    return jax.lax.conv_general_shape_tuple(
        lhs,
        rhs,
        strides,
        padding,
        ('NCHW', 'HWIO', 'NCHW'),
    )


def max_pool_output_shape(lhs, pool_size, strides, padding):
    return jax.lax.reduce_window_shape_tuple(
        lhs,
        pool_size,
        strides,
        jax.lax.padtype_to_pads(
            lhs,
            pool_size,
            strides,
            padding,
        ),
    )


@partial(jax.jit, static_argnames=('strides', 'padding'))
def conv2d(inputs, W, b, strides, padding):
    return jax.lax.conv_general_dilated(
        inputs,
        W,
        strides,
        padding,
        dimension_numbers=('NCHW', 'HWIO', 'NCHW'),
    ) + b[None, :, None, None]


@partial(jax.jit, static_argnames=('pool_size', 'strides', 'padding'))
def max_pool(inputs, pool_size, strides, padding):
    window_shape = (1, 1, pool_size[0], pool_size[1])
    stride_shape = (1, 1,   strides[0],   strides[1])
    return jax.lax.reduce_window(
        inputs,
        -jnp.inf,
        jax.lax.max,
        window_shape,
        stride_shape,
        padding,
    )


@jax.jit
def flatten(inputs):
    return inputs.reshape(inputs.shape[0], -1)


@jax.jit
def dense(inputs, W, b):
    return inputs @ W + b


def conv2d_init_layer_params(kernel_size, input_channel, output_channel, key):
    shape = (kernel_size, kernel_size, input_channel, output_channel)  # HWIO
    W = jax.nn.initializers.glorot_normal(2, 3)(key, shape)
    b = jnp.zeros((output_channel,), dtype=float)
    return W, b


def dense_init_layer_params(input_dim, output_dim, key):
    shape = (input_dim, output_dim)
    W = jax.nn.initializers.glorot_normal()(key, shape)
    b = jnp.zeros((output_dim,), dtype=float)
    return W, b


LAYERS = {}


def register(cls):
    LAYERS[cls.__name__] = cls
    return cls


@register
class InputLayer(NamedTuple):
    shape: tuple[int | None, int, int, int]


@register
class Conv2DLayer(NamedTuple):
    channels: int
    kernel_size: tuple[int, int]
    strides: tuple[int, int]
    padding: str
    activation: str


@register
class MaxPoolLayer(NamedTuple):
    pool_size: tuple[int, int]
    strides: tuple[int, int]
    padding: str


@register
class FlattenLayer(NamedTuple):
    pass


@register
class DenseLayer(NamedTuple):
    units: int
    activation: str


def encode_layers(layers):
    return [
        {'class': layer.__class__.__name__, **layer._asdict()}
        for layer in layers
    ]


def decode_layers(data):
    layers = list()
    for spec in data:
        genre = spec.pop('class')
        cls = LAYERS.get(genre)
        if cls is None:
            raise ValueError(f"unknown layer: {genre}")
        layers.append(cls(**spec))
    return tuple(layers)


def cnn_init_network_params(layers: tuple, key):
    params = list()
    keys = jax.random.split(key, len(layers))
    shape = layers[0].shape
    shapes = [shape]
    for j, layer in enumerate(layers[1:]):
        match layer:
            case Conv2DLayer(
                kernel_size=(h, w),
                channels=o,
                strides=st,
                padding=pd,
            ):
                i = shape[1]
                W, b = conv2d_init_layer_params(h, i, o, keys[j])
                params.append({'W': W, 'b': b})
                shape = conv2d_output_shape(shape, (h, w, i, o), st, pd)
            case MaxPoolLayer(
                pool_size=(h, w),
                strides=(s, t),
                padding=pd,
            ):
                params.append({})
                shape = (None, ) + max_pool_output_shape(
                    shape[1:],
                    (1, h, w),
                    (1, s, t),
                    pd,
                )
            case FlattenLayer():
                params.append({})
                shape = (shape[0], np.prod(shape[1:]))
            case DenseLayer(units=units):
                W, b = dense_init_layer_params(shape[1], units, keys[j])
                params.append({'W': W, 'b': b})
                shape = (None, units)
            case _:
                raise ValueError(f"unknown layer: {layer}")
        shapes.append(shape)
    print('cnn_init_network_params', shapes)
    return params


@partial(jax.jit, static_argnames=('layers',))
def cnn_forward_batch(params, layers, inputs):
    x = inputs
    for param, layer in zip(params, layers[1:]):
        match layer:
            case Conv2DLayer(strides=st, padding=pd, activation=act):
                x = conv2d(x, param['W'], param['b'], st, pd)
                if act == 'relu':
                    x = relu(x)
                else:
                    raise ValueError(f"unknown activation: {act}")
            case MaxPoolLayer(pool_size=ps, strides=st, padding=pd):
                x = max_pool(x, ps, st, pd)
            case FlattenLayer():
                x = flatten(x)
            case DenseLayer(activation=act):
                x = dense(x, param['W'], param['b'])
                if act == 'relu':
                    x = relu(x)
                elif act == 'sigmoid':
                    x = sigmoid(x)
                else:
                    raise ValueError(f"unknown activation: {act}")
            case _:
                raise ValueError(f"unknown layer: {layer}")
    return x


@partial(jax.jit, static_argnames=('layers',))
def cnn_forward(params, layers, intake):
    inputs = jnp.expand_dims(intake, axis=0)
    outputs = cnn_forward_batch(params, layers, inputs)
    return jnp.squeeze(outputs, axis=0)


def step(params, grads, alpha):
    return jax.tree.map(
        lambda param, grad: param - alpha * grad,
        params,
        grads,
    )
