# G5
[![Language](https://img.shields.io/github/languages/top/zackxzhang/G5)](https://github.com/zackxzhang/G5)
[![License](https://img.shields.io/github/license/zackxzhang/G5)](https://opensource.org/licenses/BSD-3-Clause)
[![Last Commit](https://img.shields.io/github/last-commit/zackxzhang/G5)](https://github.com/zackxzhang/G5)

*deep reinforcement learning for gomoku*


##### scaffolding
- [x] state transition
- [ ] user interface

##### deep learning
- [x] value network
- [x] policy network
- [ ] rate scheduling
- [ ] weight averaging
- [ ] gradient smoothing

##### reinforcement learning
- [x] experience replay
- [x] self reinforcement
- [ ] importance sampling
- [ ] reward shaping
- [ ] prior injection

##### tree search
- [ ] minimax
- [ ] monte carlo

##### match result
|       player     || win ratio                    |||
|:-------:|:-------:|---------:|---------:|---------:|
|   `X`   |   `O`   |    `X`   |    `-`   |    `O`   |
| random  | random  |        % |        % |        % |
|   agent | random  |        % |        % |        % |
| random  | agent   |        % |        % |        % |
|   agent | agent   |        % |        % |        % |
