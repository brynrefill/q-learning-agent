"""
Deep Q-Network (DQN) for CliffWalking environment
"""
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

CHECKPOINT  = "./checkpoints/dqn.weights.pth"
os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)

# set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# HYPERPARAMETERS -----------------------------
N_EPISODES      = 500 # if is_det else 1000
N_TEST_EPISODES = 5
MAX_N_STEPS     = 200 # if is_det else 500

# alpha / step size
LEARNING_RATE = 0.001 # if is_det else 0.0005

# gamma
DISC_FACTOR = 0.9 # if is_det else 0.95

# epsilon
EXPL_RATE_MAX = 1.0
EXPL_RATE_MIN = 0.01 # if is_det else 0.05
EXPL_RATE     = EXPL_RATE_MAX
EPSILON_DECAY = 0.995

# experience
BUFFER_SIZE = 10000 # if is_det else 20000
BATCH_SIZE  = 64
replay_buffer = deque(maxlen=BUFFER_SIZE)
# ---------------------------------------------

# CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model_weights(model):
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"Model weights saved.")

def restore_model_weights(model):
    model.load_state_dict(torch.load(CHECKPOINT))
    print(f"Model weights restored.")
    
    # put the model in evaluation (inference) mode
    model.eval() # ??

# (dense) neural network construction
def create_q_network(is_trained, n_states, n_actions):

    # INPUT LAYER: vector with 48 features per sample, representing the state
    # FIRST and SECOND HIDDEN LAYERS : 32 hidden units (neurons) and ReLU activation function (that introduces non-linearity, so the model can learn complex patterns)
    # OUTPUT LAYER: vector with 4 floats, for action Q-values

    model = nn.Sequential(
        nn.Linear(n_states, 32),
        nn.ReLU(),

        nn.Linear(32, 32),
        nn.ReLU(),

        nn.Linear(32, n_actions)
    )

    if is_trained: restore_model_weights(model)

    return model.to(device)

# convert discrete state to one-hot encoded vector (1D tensor/vector of 48 integers)
def state_to_tensor(state, n_states):
    one_hot = torch.zeros(n_states, device=device)
    one_hot[state] = 1.0
    return one_hot

# next action selection
def epsilon_greedy_policy(q_network, state, n_states, n_actions, expl_rate):
    if random.random() < expl_rate:
        # exploration: random action
        return random.randint(0, n_actions - 1)
    else:
        # exploitation: greedy action
        # disable gradient computation, for only inference (not training)
        with torch.no_grad():
            state_tensor = state_to_tensor(state, n_states).unsqueeze(0)

            # action Q-values
            q_values = q_network(state_tensor)

            # action related to highest Q-value
            return q_values.argmax().item()

# perform one training step on a batch of experiences
def train_step(q_network, optimizer, batch, n_states, n_actions):

    # unpack batch
    states, actions, rewards, next_states, terms = zip(*batch)

    # convert the unpacked lists to tensors
    states_tensor = torch.stack([state_to_tensor(s, n_states) for s in states])
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_tensor = torch.stack([state_to_tensor(s, n_states) for s in next_states])
    terms_tensor = torch.tensor(terms, dtype=torch.float32, device=device)

    '''
    print(states_tensor)
    print(actions_tensor)
    print(rewards_tensor)
    print(next_states_tensor)
    print(terms_tensor)

    exit()
    '''

    # compute current Q-values
    q_current = q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    
    # compute target (next) Q-values
    with torch.no_grad():
        q_next = q_network(next_states_tensor).max(1)[0]

        # update weights using the Bellman's equation: Q(s,a) = r + gamma * max(Q(s',a'))
        target_q_values = rewards_tensor + DISC_FACTOR * q_next * (1 - terms_tensor)


    # compute MSE loss: how far off the model's predictions are from the targets
    loss = nn.MSELoss()(q_current, target_q_values)
    
    # optimize the network
    # clear previous gradients from the previous backward pass: in PyTorch,
    # gradients accumulate by default
    optimizer.zero_grad()

    # backpropagation: compute new gradients of the loss wrt the weights.
    # PyTorch builds the computational graph and applies the chain rule
    loss.backward()

    # update weights: uses the gradients computed to update the weights of the model.
    # The specific update rule depends on the optimizer (e.g. SGD, Adam)
    optimizer.step()
    
    return loss.item()

# test the trained agent
def test_agent(env):
    # environment
    N_STATES = env.observation_space.n
    N_ACTIONS = env.action_space.n

    print(f"Testing the agent over {N_TEST_EPISODES} episodes...")

    # network
    is_trained = True
    q_network = create_q_network(is_trained, N_STATES, N_ACTIONS)

    ep_rewards = []

    for episode in range(N_TEST_EPISODES):
        state, _ = env.reset()
        terminated = False
        steps = 0
        reward_sum = 0

        while not terminated and steps < MAX_N_STEPS:
            # choose the action using the optimal policy
            action = epsilon_greedy_policy(q_network, state, N_STATES, N_ACTIONS, 0.0)

            # do the action
            next_state, reward, terminated, _, _ = env.step(action)

            # update the state
            state = next_state

            reward_sum += reward
            steps += 1

        ep_rewards.append(reward_sum)
        print(f"[{episode + 1}/{N_TEST_EPISODES}] Reward: {reward_sum}, steps: {steps}")

    env.close()
    avg_ep_rewards = np.mean(ep_rewards)
    print(f"Average test reward: {avg_ep_rewards:.2f}.")
    print(f"Testing completed!")

# train the agent with Deep Q-Network
def q_learning(env):
    global EXPL_RATE

    # environment
    # 4 * 12 possible tiles in the env grid
    N_STATES = env.observation_space.n
    # 4 possible actions: up, down, right and left
    N_ACTIONS = env.action_space.n

    # Q-Network and optimizer
    is_trained = False
    q_network = create_q_network(is_trained, N_STATES, N_ACTIONS)
    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    # lists for statistics
    ep_rewards = deque(maxlen=100)
    ep_steps = deque(maxlen=100)
    avg_ep_rewards = []
    avg_ep_steps = []

    print(f"Training the agent (device: {device}) with Deep Q-Network over {N_EPISODES} episodes...")
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
            action = epsilon_greedy_policy(q_network, state, N_STATES, N_ACTIONS, EXPL_RATE)

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

            # store the experience into replay buffer
            # (it replays the transitions seen in the past)
            replay_buffer.append((state, action, reward, next_state, terminated))

            # update the state
            state = next_state

            reward_sum += reward
            steps += 1
            
            # train
            if len(replay_buffer) >= BATCH_SIZE:
                # sample a batch of experiences from replay buffer
                batch = random.sample(replay_buffer, BATCH_SIZE)

                loss = train_step(q_network, optimizer, batch, N_STATES, N_ACTIONS)

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
    save_model_weights(q_network)
    env.close()
    print(f"Training completed in {training_duration:.2f} seconds!\n")

    return avg_ep_rewards
