from typing import Any, TypeAlias, NamedTuple
from jax import Array                                             # type: ignore
from jaxtyping import Int


Stone:  TypeAlias = int
Board:  TypeAlias = Int[Array, '15 15']
Boards: TypeAlias = Int[Array, 'N 15 15']
Point:  TypeAlias = int
Points: TypeAlias = Int[Array, 'N']
Coord:  TypeAlias = Int[Array, '1 2']
Coords: TypeAlias = Int[Array, 'N 1 2']
Action: TypeAlias = tuple[Stone, Point]


PyTree: TypeAlias = Any
Key:    TypeAlias = Array


Layer:  TypeAlias = NamedTuple
Layers: TypeAlias = tuple[Layer, ...]
