import jax                                                        # type: ignore
import unittest
from g5.agent import Amateur
from g5.reward import Victory
from g5.game import Game, Score


class GameTestCase(unittest.TestCase):

    def test_play(self):
        p1 = Amateur(
            stone=+1,
            reward=Victory,
            key=jax.random.key(1),
        )
        p2 = Amateur(
            stone=-1,
            reward=Victory,
            key=jax.random.key(2),
        )
        score = Score()
        for _ in range(5):
            game = Game((p1, p2))
            while True:
                agent  = game.agent
                action = agent.act(game.board)
                winner = game.evo(action)
                if winner in (-1, 0, +1):
                    score(winner)
                    print(f"The game took {len(game)} steps.")
                    break
        print(score)
