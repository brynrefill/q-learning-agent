import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time

def save_q_table(q_table):
    try:
        np.save('./checkpoints/q_table.npy', q_table)
        print("Q-table saved in ./checkpoints/q_table.npy.")
    except FileNotFoundError:
        print("Missing the ./checkpoints folder. Q-table is not saved.")

def restore_q_table():
    try:
        q_table = np.load('./checkpoints/q_table.npy')
        print(q_table)
        print("Q-table restored from ./checkpoints/q_table.npy.")
    except FileNotFoundError:
        q_table = None
        print("You must train the agent first. Q-table is not restored.")
    return q_table

def epsilon_greedy_policy(env, q_table, state, expl_rate):
    if np.random.random() < expl_rate:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

# test the agent
def q_testing(q_table, is_det):
    env = gym.make('CliffWalking-v0', is_slippery=(not is_det), render_mode="human")

    print(f"Testing phase...")
    state, _ = env.reset()
    terminated = False
    steps = 0
    reward_sum = 0

    while not terminated:
        # choose the action using the optimal policy
        action = epsilon_greedy_policy(env, q_table, state, 0.0)

        # do the action
        next_state, reward, terminated, _, _ = env.step(action)

        # update the state
        state = next_state

        steps +=1
        reward_sum += reward

    env.close()
    print(f"[1/1] Reward: {reward_sum}, Steps: {steps}")
    print(f"Testing phase completed.")

# train the agent
def q_learning(is_det):
    env = gym.make('CliffWalking-v0', is_slippery=(not is_det))
    NUM_STATES = env.observation_space.n # 4 * 12 possible tiles in the env grid
    NUM_ACTIONS = env.action_space.n     # 4 possible actions: up, down, right and left
    NUM_EPISODES = 1000

    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
    ep_rewards = []
    ep_steps = []
    ep_expl_rates = []

    # step size / alpha
    LEARNING_RATE = 1.0 if is_det else 0.1

    # gamma
    DISC_FACTOR = 0.9

    # epsilon
    EXPL_RATE_MAX = 1.0
    EXPL_RATE_MIN = 0.01
    EXPL_RATE = EXPL_RATE_MAX
    # EXPL_DECAY_FACTOR = (EXPL_RATE_MAX - EXPL_RATE_MIN) / NUM_EPISODES      # == 0.00099 ~ 0.001 linear decay
    EXPL_DECAY_FACTOR = (EXPL_RATE_MIN / EXPL_RATE_MAX) ** (1 / NUM_EPISODES) # == 0.995 exponential decay

    print(f"Training phase with Tabular Q-learning over {NUM_EPISODES} episodes...")
    start_time = time.time()
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        terminated = False
        steps = 0
        reward_sum = 0

        while not terminated:
            # choose the action using the epsilon-greedy policy
            action = epsilon_greedy_policy(env, q_table, state, EXPL_RATE)

            # do the action
            next_state, reward, terminated, _, _ = env.step(action)

            # update the Q-table using the Bellman's equation
            q_table[state][action] = (1 - LEARNING_RATE) * q_table[state][action] + LEARNING_RATE * (reward + DISC_FACTOR * np.max(q_table[next_state]))

            # update the state
            state = next_state

            steps +=1
            reward_sum += reward

        # save information for later evaluations
        ep_steps.append(steps)
        ep_rewards.append(reward_sum)
        ep_expl_rates.append(EXPL_RATE)

        # update the exploration rate to decrease the exploration:exploitation ratio
        # linear epsilon decrease
        # EXPL_RATE = max(EXPL_RATE - EXPL_DECAY_FACTOR, EXPL_RATE_MIN) # EXPL_RATE = EXPL_RATE - EXPL_DECAY_FACTOR
        # exponential epsilon decrease
        EXPL_RATE = max(EXPL_RATE * EXPL_DECAY_FACTOR, EXPL_RATE_MIN)

        if (episode + 1) % 25 == 0:
            print(f"[{episode + 1}/{NUM_EPISODES}] Reward: {reward_sum}, Steps: {steps}")

    training_duration = time.time() - start_time
    env.close()
    print(f"Training phase completed in {training_duration:.2f} seconds.")

    return q_table, ep_rewards, ep_steps, ep_expl_rates
