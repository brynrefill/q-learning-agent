from tabular_q_learning import *

def main():
    # change to False to test the already trained agent
    to_train = True

    # change to False to make the environment stochastic
    # where the agent will move in the intended direction with probability == 1/3
    is_det = True

    if (to_train):
        q_table, q_rewards, q_steps, q_expl_rates = q_learning(is_det)
        save_q_table(q_table)
    else:
        q_table = restore_q_table()

    if(q_table is not None): q_testing(q_table, is_det)

if __name__ == "__main__":
    main()
