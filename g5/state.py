import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
import jax.scipy as jsp                                           # type: ignore
from jax import Array                                             # type: ignore
from jax.typing import ArrayLike                                  # type: ignore
from jaxtyping import Int, Float
from typing import TypeAlias


Stone: TypeAlias = int
Board: TypeAlias = Int[Array, '15 15']
Coord: TypeAlias = Int[Array, '1 2']
Action: TypeAlias = tuple[Stone, Coord]


onset = jnp.zeros((15, 15), dtype=jnp.int32)
proxy = jnp.zeros((2,), dtype=jnp.int32)


def _stringify(stone):
    match stone:
        case 1:
            return 'X'
        case -1:
            return 'O'
        case 0:
            return '_'
        case _:
            raise ValueError(f'unrecognized stone {_}')


def stringify(board: Board) -> str:
    return '\n'.join(''.join(map(_stringify, row)) for row in board)


def affordance(board: Board) -> list[Coord]:
    return jnp.argwhere(board == 0)


def transition(board: Board, stone: Stone, coord: Coord) -> Board:
    return board.at[coord[0], coord[1]].set(stone)


def unravel(index: int) -> Coord:
    i, j = divmod(index, 15)
    return jnp.array([i, j])


transitions = jax.vmap(transition, in_axes=(None, None, 0))


kernels = [
    jnp.ones((1, 5)),
    jnp.ones((5, 1)),
    jnp.eye(5),
    jnp.fliplr(jnp.eye(5)),
]


def victorious(board: Board) -> int:
    for kernel in kernels:
        points = jsp.signal.convolve2d(board, kernel, mode='valid')
        if (points == 5).any():
            return 1
        elif (points == -5).any():
            return -1
    return 0


def impasse(board: Board) -> bool:
    return (board == 0).sum() == 0


def judge(board: Board) -> Stone:
    victor = victorious(board)
    if victor != 0:
        return victor
    if impasse(board):
        return 2
    return 0
