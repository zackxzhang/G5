import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
import jax.scipy as jsp                                           # type: ignore
from jax import Array                                             # type: ignore
from jaxtyping import Int
from typing import TypeAlias


Stone: TypeAlias = int
Board: TypeAlias = Int[Array, '15 15']
Coord: TypeAlias = Int[Array, '1 2']
Coords: TypeAlias = Int[Array, 'B 1 2']
Action: TypeAlias = tuple[Stone, Coord]


onset = jnp.zeros((15, 15), dtype=int)
proxy = jnp.zeros((2,), dtype=int)


def _stringify(stone):
    match stone:
        case 1:
            return 'X'
        case -1:
            return 'O'
        case 0:
            return '_'
        case _:
            raise ValueError(f'unrecognized stone {stone}')


def stringify(board: Board) -> str:
    return '\n'.join(''.join(map(_stringify, row)) for row in board)


def affordance(board: Board) -> Coords:
    return jnp.argwhere(board == 0)


def transition(board: Board, stone: Stone, coord: Coord) -> Board:
    return board.at[coord[0], coord[1]].set(stone)


def unravel(index: int) -> Coord:
    i, j = divmod(index, 15)
    return jnp.array([i, j])


transitions = jax.jit(jax.vmap(transition, in_axes=(None, None, 0)))


kernels = [
    jnp.ones((1, 5), dtype=float),
    jnp.ones((5, 1), dtype=float),
    jnp.eye(5, dtype=float),
    jnp.fliplr(jnp.eye(5, dtype=float)),
]


@jax.jit
def conv(x, y):
    return jsp.signal.convolve2d(x, y, mode='valid')


def victorious(board: Board) -> int:
    for kernel in kernels:
        points = conv(board, kernel)
        if (points == 5).any():
            return 1
        elif (points == -5).any():
            return -1
    return 0


@jax.jit
def impasse(board: Board):
    return (board == 0).sum() == 0


def judge(board: Board) -> Stone:
    victor = victorious(board)
    if victor != 0:
        return victor
    if impasse(board):
        return 0
    return 9
