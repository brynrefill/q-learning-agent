'''
Note: the CliffWalking-v0 environment could have been solved with a simple DQN.
However, I chose to use a Double DQN for a more complete implementation.
This approach uses two networks to address two key issues:
1. reducing training instability by providing fixed Q-value targets during updates;
2. reducing overestimation of Q-values by decoupling action selection from action evaluation.
'''
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import time

CHECKPOINT  = "./checkpoints/dqn.weights.h5"
os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)

def save_model_weights(model):
    model.save_weights(CHECKPOINT)
    print(f"Model weights saved in {CHECKPOINT}.")

def restore_model_weights(model):
    model.load_weights(CHECKPOINT)
    print(f"Model weights restored from {CHECKPOINT}.")

def update_target_network(model, target_model):
    target_model.set_weights(model.get_weights())

# DQN construction with embedding
def build_dqn(is_trained, n_states, n_actions, learning_rate=0.001):
    state_input = Input(shape=(1,), dtype=tf.int32)
    x = Embedding(input_dim=n_states, output_dim=16)(state_input)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    q_output = Dense(n_actions, activation='linear')(x)

    model = Model(inputs=state_input, outputs=q_output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    if is_trained: restore_model_weights(model)
    return model

def epsilon_greedy_policy(env, model, state, expl_rate):
    if np.random.random() < expl_rate:
        return env.action_space.sample()
    else:
        q_values = model(np.array([[state]]), training=False).numpy()[0]
        return np.argmax(q_values)

# test the agent
def q_testing(is_det):
    env = gym.make('CliffWalking-v0', is_slippery=(not is_det), render_mode="human")
    NUM_STATES  = env.observation_space.n
    NUM_ACTIONS = env.action_space.n

    model = build_dqn(True, NUM_STATES, NUM_ACTIONS)

    print(f"Testing phase...")
    state, _ = env.reset()
    terminated = False
    steps = 0
    reward_sum = 0

    while not terminated:
        # choose the action using the optimal policy
        action = epsilon_greedy_policy(env, model, state, 0.0)

        # do the action
        next_state, reward, terminated, _, _ = env.step(action)

        # update the state
        state = next_state

        steps += 1
        reward_sum += reward

    env.close()
    print(f"[1/1] Reward: {reward_sum}, Steps: {steps}")
    print(f"Testing phase completed.")

# train the agent with Deep Q-learning (DDQN)
def q_learning(is_det):
    env = gym.make('CliffWalking-v0', is_slippery=(not is_det))
    NUM_STATES  = env.observation_space.n # 4 * 12 possible tiles in the env grid
    NUM_ACTIONS = env.action_space.n      # 4 possible actions: up, down, right and left
    NUM_EPISODES  = 1000 if is_det else 2000
    MAX_EPISODE_STEPS = 1000

    # step size / alpha
    LEARNING_RATE = 0.001 if is_det else 0.0005 # here (wrt the tabular case), lr=1 is bad because it makes our Q-value updates overwrite completely the old estimate, ignoring all past learning

    # gamma
    DISC_FACTOR = 0.9

    # epsilon
    EXPL_RATE_MAX = 1.0
    EXPL_RATE_MIN = 0.01
    EXPL_RATE = EXPL_RATE_MAX
    EXPL_DECAY_FACTOR = (EXPL_RATE_MAX - EXPL_RATE_MIN) / NUM_EPISODES

    # neural networks
    model = build_dqn(False, NUM_STATES, NUM_ACTIONS, LEARNING_RATE)
    target_model = build_dqn(False, NUM_STATES, NUM_ACTIONS, LEARNING_RATE)
    update_target_network(model, target_model)
    TARGET_UPDATE_FREQ = 5 if is_det else 10 # to avoid chasing random fluctuations

    # replay buffer
    EXP_MAX_SIZE = 5000 if is_det else 10000
    BATCH_SIZE = 64 if is_det else 128 # (EXP_MAX_SIZE/BATCH_SIZE) == unique possible updates before the batches start repeating
    experience = deque(maxlen=EXP_MAX_SIZE)

    ep_rewards = []
    ep_steps = []
    ep_expl_rates = []

    print(f"Training phase with Deep Q-learning (DDQN) over {NUM_EPISODES} episodes...")
    start_time = time.time()
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        terminated = False
        steps = 0 # timesteps
        reward_sum = 0

        while not terminated and steps < MAX_EPISODE_STEPS:
            # choose the action using the epsilon-greedy policy
            action = epsilon_greedy_policy(env, model, state, EXPL_RATE)

            # do the action
            next_state, reward, terminated, _, _ = env.step(action)

            # save the experience in the replay buffer
            experience.append((state, action, reward, next_state, terminated))

            # update the state
            state = next_state

            steps += 1
            reward_sum += reward

        # save information for later evaluations
        ep_steps.append(steps)
        ep_rewards.append(reward_sum)
        ep_expl_rates.append(EXPL_RATE)

        # train the DQN
        if len(experience) >= BATCH_SIZE:
            minibatch = random.sample(experience, BATCH_SIZE)
            states, actions, rewards, next_states, terms = zip(*minibatch)

            states = np.array(states).reshape(-1, 1)
            next_states = np.array(next_states).reshape(-1, 1)

            q_next_online = model(next_states, training=False).numpy()        # Q-online
            q_next_target = target_model(next_states, training=False).numpy() # Q-target
            targets = model(states, training=False).numpy()

            for i in range(BATCH_SIZE):
                if terms[i]:
                    target = rewards[i]
                else:
                    best_action = np.argmax(q_next_online[i])
                    target = rewards[i] + DISC_FACTOR * q_next_target[i][best_action]
                targets[i][actions[i]] = target

            model.fit(states, targets, epochs=1, verbose=0)

        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            update_target_network(model, target_model)
            print(f"Target network updated at episode {episode + 1}.")

        # update the exploration rate to decrease the exploration:exploitation ratio
        EXPL_RATE = max(EXPL_RATE - EXPL_DECAY_FACTOR, EXPL_RATE_MIN) # linear epsilon decrease

        print(f"[{episode + 1}/{NUM_EPISODES}] Epsilon: {EXPL_RATE:.3f}, Reward: {reward_sum}, Steps: {steps}")
        if (episode + 1) % 100 == 0: save_model_weights(model)

    training_duration = time.time() - start_time
    env.close()
    print(f"Training phase completed in {training_duration:.2f} seconds.")

    return ep_rewards, ep_steps, ep_expl_rates
