import tabular_q_learning as tql
import deep_q_network as dqn
import gymnasium as gym

# change to False to test the already trained agent
TO_TRAIN = True

# change to True to make the environment stochastic
# where the agent will move in the intended direction with probability == 1/3
IS_NOT_DET = False

def create_env(to_train=True):
    render_mode = None if to_train else "human"
    env = gym.make('CliffWalking-v1', is_slippery=IS_NOT_DET, render_mode=render_mode)
    return env

def main():

    # lists for later evaluations
    ep_rewards = []
    avg_ep_rewards = []
    avg_ep_steps = []

    print("x----------------------------------------------------------------------+")
    print("| Study of a Q-learning agent in the Cliff Walking environment.        |")
    print("o----------------------------------------------------------------------/")

    # ----------------------- Tabular Q-learning train and test --------------------

    # training
    if (TO_TRAIN):
        avg_ep_rewards, avg_ep_steps = tql.q_learning(create_env())

    # testing
    test_ep_rewards, test_avg_ep_rewards = tql.test_agent(create_env(not TO_TRAIN))

    print("Tabular Q-learning training and testing completed successfully!")
    print("========================================================================")

    # ----------------------- Deep Q-Network train and test ------------------------

    # training
    if (TO_TRAIN):
        avg_ep_rewards, avg_ep_steps = dqn.q_learning(create_env())

    # testing
    test_ep_rewards, test_avg_ep_rewards = dqn.test_agent(create_env(not TO_TRAIN))

    print("Deep Q-Network training and testing completed successfully!")
    print("========================================================================")

    '''
    # TODO: evaluations
    # ...
    '''

    print("x----------------------------------------------------------------------+")
    print("| The end.                                                             |")
    print("o----------------------------------------------------------------------/")

if __name__ == "__main__":
    main()
