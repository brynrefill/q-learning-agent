"""
Tabular Q-learning for CliffWalking environment
"""
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time

CHECKPOINT = "./checkpoints/q_table.npy"
os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)

# set seed for reproducibility
random.seed(42)
np.random.seed(42)

# HYPERPARAMETERS -----------------------------
N_EPISODES      = 500 # if is_det else 1000
N_TEST_EPISODES = 5
MAX_N_STEPS     = 200 # if is_det else 500

# alpha / step size
LEARNING_RATE = 1.0   # if is_det else 0.1

# gamma
DISC_FACTOR = 0.9     # if is_det else 0.95

# epsilon
EXPL_RATE_MAX = 1.0
EXPL_RATE_MIN = 0.01  # if is_det else 0.05
EXPL_RATE     = EXPL_RATE_MAX
EPSILON_DECAY = 0.995
# ---------------------------------------------

def save_q_table(q_table):
    np.save(CHECKPOINT, q_table)
    print(f"Q-Table saved.")

def restore_q_table():
    q_table = np.load(CHECKPOINT)
    # print(q_table)
    print(f"Q-Table restored.")
    return q_table

def create_q_table(is_trained, n_states, n_actions):
    if is_trained:
        return restore_q_table()
    return np.zeros((n_states, n_actions))

# next action selection
def epsilon_greedy_policy(q_table, state, n_actions, expl_rate):
    if np.random.random() < expl_rate:
        # exploration: random action
        return np.random.randint(0, n_actions - 1)
    else:
        # exploitation: greedy action
        return np.argmax(q_table[state])

# test the trained agent
def test_agent(env):
    # environment
    N_STATES = env.observation_space.n
    N_ACTIONS = env.action_space.n

    print(f"Testing the agent over {N_TEST_EPISODES} episodes...")

    # table
    is_trained = True
    q_table = create_q_table(is_trained, N_STATES, N_ACTIONS)

    ep_rewards = []

    for episode in range(N_TEST_EPISODES):
        state, _ = env.reset()
        terminated = False
        steps = 0
        reward_sum = 0

        while not terminated and steps < MAX_N_STEPS:
            # choose the action using the optimal policy
            action = epsilon_greedy_policy(q_table, state, N_ACTIONS, 0.0)

            # do the action
            next_state, reward, terminated, _, _ = env.step(action)

            # update the state
            state = next_state

            reward_sum += reward
            steps +=1

        ep_rewards.append(reward_sum)
        print(f"[{episode + 1}/{N_TEST_EPISODES}] Reward: {reward_sum}, steps: {steps}")

    env.close()
    avg_ep_rewards = np.mean(ep_rewards)
    print(f"Average test reward: {avg_ep_rewards:.2f}.")
    print(f"Testing completed!")

# train the agent with Tabular Q-learning
def q_learning(env):
    global EXPL_RATE

    # environment
    # 4 * 12 possible tiles in the env grid
    N_STATES = env.observation_space.n
    # 4 possible actions: up, down, right and left
    N_ACTIONS = env.action_space.n

    # Q-Table
    is_trained = False
    q_table = create_q_table(is_trained, N_STATES, N_ACTIONS)

    # lists for statistics
    ep_rewards = deque(maxlen=100)
    ep_steps = deque(maxlen=100)
    avg_ep_rewards = []
    avg_ep_steps = []

    print(f"Training the agent with Tabular Q-learning over {N_EPISODES} episodes...")
    start_time = time.time()

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        terminated = False

        # timesteps
        steps = 0

        # cumulative reward of an episode
        reward_sum = 0

        while not terminated and steps < MAX_N_STEPS:
            # choose the action using the epsilon-greedy policy
            action = epsilon_greedy_policy(q_table, state, N_ACTIONS, EXPL_RATE)

            # do the action
            # in this env 'truncated' seems that is not used
            next_state, reward, terminated, _, _ = env.step(action)

            '''
            print(state)
            print(action)
            print(reward)
            print(next_state)
            print(terminated)
            print(truncated)

            exit()
            '''

            # update the Q-Table using the Bellman's equation
            q_table[state][action] = (1 - LEARNING_RATE) * q_table[state][action] + LEARNING_RATE * (reward + DISC_FACTOR * np.max(q_table[next_state]))

            # update the state
            state = next_state

            reward_sum += reward
            steps +=1

        # decay epsilon to decrease the exploration:exploitation ratio
        # exponential epsilon decrease
        EXPL_RATE = max(EXPL_RATE * EPSILON_DECAY, EXPL_RATE_MIN)

        # save information for later evaluations
        ep_rewards.append(reward_sum)
        ep_steps.append(steps)

        # calculate moving avg over the last 100 episodes
        avg_reward = np.mean(ep_rewards)
        avg_steps = round(np.mean(ep_steps))
        avg_ep_rewards.append(avg_reward)
        avg_ep_steps.append(avg_steps)

        # log progress every 50 episodes
        if (episode + 1) % 50 == 0:
            print(f"[{episode + 1}/{N_EPISODES}] "
                    f"avg steps: {avg_steps}, "
                    f"avg reward: {avg_reward:.2f}, "
                    f"epsilon: {EXPL_RATE:.3f}")

    training_duration = time.time() - start_time
    save_q_table(q_table)
    env.close()
    print(f"Training completed in {training_duration:.2f} seconds!\n")

    return avg_ep_rewards
