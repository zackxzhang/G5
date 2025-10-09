import jax                                                        # type: ignore
import jax.numpy as jnp                                           # type: ignore
import jax.scipy as jsp                                           # type: ignore
from flax.struct import dataclass, field                          # type: ignore
from .hint import Stone, Board, Point, Coord, Coords


onset = jnp.zeros((15, 15), dtype=int)
proxy = jnp.zeros((2,), dtype=int)


def _stringify(stone: Stone):
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


def ravel(coord: Coord) -> Point:
    x, y = coord
    return x * 15 + y


def unravel(point: Point) -> Coord:
    x, y = divmod(point, 15)
    return jnp.array([x, y])


@dataclass
class Affordance:
    points: jnp.ndarray = field(default_factory=lambda: jnp.arange(225))
    locator: jnp.ndarray = field(default_factory=lambda: jnp.arange(225))
    n: int = 225


@jax.jit
def remove(aff: Affordance, point: Point) -> Affordance:
    loc = aff.locator[point]
    last = aff.points[aff.n - 1]
    points = aff.points.at[loc].set(last)
    locator = aff.locator.at[last].set(loc).at[point].set(-1)
    return Affordance(points, locator, aff.n - 1)


@jax.jit
def transition(board: Board, stone: Stone, point: Point) -> Board:
    x, y = jnp.divmod(point, 15)
    return board.at[x, y].set(stone)


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
