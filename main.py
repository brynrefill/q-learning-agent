import argparse
import deep_q_network as dqn
import gymnasium as gym
import numpy as np
import os
import q_learning_plot as g
import sys
import tabular_q_learning as tql

def print_usage(message):
    print(f"Usage: python3 main.py [-h] --env ENV_TYPE\n{message}")
    sys.exit(1)

# create the parser
parser = argparse.ArgumentParser(description="")
parser.error = print_usage

# add argument for environment type and parse the arguments
parser.add_argument(
    "--env",
    type=str,
    required=True,
    metavar="ENV_TYPE",
    help="environment type. Enter 'det' for deterministic, 'ndet' for non-deterministic"
)

args = parser.parse_args()
env_input = args.env.strip().lower()

# if is_slippery=True, the environment becomes stochastic (non-deterministic)
# where the agent will move in the intended direction with probability == 1/3
if env_input == "det":
    is_slippery = False
elif env_input == "ndet":
    is_slippery = True
else:
    print_usage("invalid environment type: enter 'det' or 'ndet'")

# state that the agent should be trained
TO_TRAIN = True

def save_data(data, alg):
    DATA = "./data/rewards" + alg + ".npy"
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.save(DATA, data)
    print(f"Rewards ({alg}) saved.")

def load_data(alg):
    DATA = "./data/rewards" + alg + ".npy"
    data = np.load("./data/rewards" + alg + ".npy")
    print(f"Rewards ({alg}) loaded.")
    return data

def create_env(to_train):
    render_mode = None if to_train else "human"
    env = gym.make('CliffWalking-v1', is_slippery=is_slippery, render_mode=render_mode)
    return env

def main():
    print("x------------------------------------------------------------------------------+")
    print("| Training and testing of a Q-learning agent in the Cliff Walking environment. |")
    print("o------------------------------------------------------------------------------/")

    print(f"Environment set to: {env_input}\n")
    avg_ep_rewards = []

    # ----------------------- Tabular Q-learning training and testing --------------------

    # training
    avg_ep_rewards = tql.q_learning(create_env(TO_TRAIN))
    save_data(avg_ep_rewards, ".tql")

    # testing
    tql.test_agent(create_env(not TO_TRAIN))

    print("Tabular Q-learning training and testing completed successfully!")
    print("========================================================================")

    # ----------------------- Deep Q-Network training and testing ------------------------

    # training
    avg_ep_rewards = dqn.q_learning(create_env(TO_TRAIN))
    save_data(avg_ep_rewards, ".dqn")

    # testing
    dqn.test_agent(create_env(not TO_TRAIN))

    print("Deep Q-Network training and testing completed successfully!")
    print("========================================================================")

    # prepare the data to be plotted
    data1 = load_data(".tql")
    data2 = load_data(".dqn")

    print("Plotting the data...")
    g.plot_data(data1, data2)

    print("x----------------------------------------------------------------------+")
    print("| The end.                                                             |")
    print("o----------------------------------------------------------------------/")

if __name__ == "__main__":
    main()
