# q-learning-agent
This university project implements and compares `Tabular Q-learning` and `Deep Q-Network` on the deterministic and non-deterministic version of `Cliff Walking` environment from [**Gymnasium**](https://gymnasium.farama.org/environments/toy_text/cliff_walking/).

It mainly helped me to understand the mechanisms of the RL algorithms, but also to examine their sensitivity to hyperparameters and environment dynamics.

![Cliff Walking gif](./gifs/cliffwalking-loop.gif)

## Installation and usage
```bash
$ git clone https://github.com/brynrefill/q-learning-agent.git
$ cd q-learning-agent
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install gymnasium matplotlib torch pygame
(.venv) $ python3 main.py --env {det | ndet}
```

## Some of the plots
<img alt="TQL vs DQN plot" src="./data/det/tql_vs_dqn_0995_64.png" width="720">
<img alt="ndet TQL vs DQN plot" src="./data/ndet/ndet_tql_vs_dqn_0995_64.png" width="720">
<img alt="TQL 4 ε_decays plot" src="./data/det/tql_diff_eps/tql_different_eps.png" width="720">
<img alt="ndet DQN 4 b_sizes plot" src="./data/ndet/dqn_diff_b_sizes/ndet_dqn_different_b_sizes.png" width="720">

Additional plots, along with the collected data, can be found in the `data/` folder.
