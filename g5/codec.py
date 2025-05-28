from flax.serialization import (                                  # type: ignore
    msgpack_serialize as pack_pytree,
    msgpack_restore as unpack_pytree,
)
from jax.random import (                                          # type: ignore
    key_data as encode_key,
    wrap_key_data as decode_key
)
from .hint import PyTree, Layer, Layers


def tuples_to_lists(data: PyTree) -> PyTree:
    if isinstance(data, tuple) and hasattr(data, '_fields'):
        return type(data)(*(tuples_to_lists(v) for v in data))
    elif isinstance(data, tuple):
        return [tuples_to_lists(v) for v in data]
    elif isinstance(data, list):
        return [tuples_to_lists(v) for v in data]
    elif isinstance(data, dict):
        return {k: tuples_to_lists(v) for k, v in data.items()}
    else:
        return data


def lists_to_tuples(data: PyTree) -> PyTree:
    if isinstance(data, list):
        return tuple(lists_to_tuples(v) for v in data)
    elif isinstance(data, dict):
        return {k: lists_to_tuples(v) for k, v in data.items()}
    else:
        return data


LAYERS: dict[str, Layer] = dict()


def register(cls):
    LAYERS[cls.__name__] = cls
    return cls


def encode_layers(layers: Layers):
    return tuples_to_lists([
        {'class': layer.__class__.__name__, **layer._asdict()}
        for layer in layers
    ])


def decode_layers(data: PyTree):
    layers = list()
    data = lists_to_tuples(data)
    for spec in data:
        genre = spec.pop('class')
        cls = LAYERS.get(genre)
        if cls is None:
            raise ValueError(f"unknown layer: {genre}")
        layers.append(cls(**spec))  # type: ignore[operator]
    return tuple(layers)
