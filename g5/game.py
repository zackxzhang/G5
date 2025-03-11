from .state import Stone, Board, Coord, Action, onset, proxy, transition, judge
from .agent import Agent


class Rollout:

    def __init__(self):
        self.coords = list()
        self.rewards = list()
        self.boards = [onset]

    def append(self, coord: Coord, reward: int, board: Board):
        self.coords.append(coord)
        self.rewards.append(reward)
        self.boards.append(board)

    @property
    def last(self):
        return self.boards[-1]

    @property
    def data(self):
        m = len(self.rewards)
        return jnp.hstack(
            jnp.array([onset] + self.boards[:-2]),
            jnp.array(self.boards[:-1]),
            jnp.array(self.coords),
            jnp.array(self.rewards),
            jnp.array(self.boards[1:]),
            jnp.array([jnp.nan] * (self.m-2) + [0.] * 2),
            jnp.array([jnp.nan] * (self.m-1) + [0.] * 1),
        )


class Game:

    def __init__(self, agents: tuple[Agent, Agent]):
        self.agents = agents
        self.round  = 0
        self.winner = 0
        self.rollout = Rollout()

    @property
    def agent(self) -> Agent:
        return self.agents[self.round % 2]

    @property
    def board(self) -> Board:
        return self.rollout.last

    def evo(self, action: Action):
        if self.winner:
            raise ValueError('game over')
        stone, coord = action
        board = transition(self.board, stone, coord)
        winner = judge(board)
        reward = self.agent.eye(winner)
        self.rollout.append(coord, reward, board)
        self.winner = winner
        self.round += 1
        if winner:  # shadow tail of rollout
            self.rollout.append(proxy, self.agent.eye(winner), board)
        return winner


class Score:

    def __init__(self):
        self.wins = [0, 0, 0]

    @property
    def n(self) -> int:
        return sum(self.wins)

    def __call__(self, winner: Stone):
        if winner == 1:
            self.wins[1] += 1
        elif winner == -1:
            self.wins[2] += 1
        else:
            self.wins[0] += 1

    def __str__(self):
        return (
            f'X {self.wins[1]/self.n}\n'
            f'- {self.wins[0]/self.n}\n'
            f'O {self.wins[2]/self.n}\n'
            '--------'
        )


class Simulator:

    def __init__(self, agents: tuple[Agent, Agent]):
        self.score = Score()

    def run(self):
        game = Game(self.agents)
        while True:
            agent  = game.agent
            action = agent.act(game.board)
            winner = game.evo(action)
            if winner:
                score(winner)
                break
        return game.rollout

    def __call__(self, n: int):
        return [self.run() for _ in range(n)]
