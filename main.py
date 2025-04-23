import tabular_q_learning as tql
import deep_q_learning as dql

def main():
    # change to False to test the already trained agent
    to_train = True

    # change to False to make the environment stochastic
    # where the agent will move in the intended direction with probability == 1/3
    is_det = True

    if (to_train):
        q_table, q_rewards, q_steps, q_expl_rates = tql.q_learning(is_det)
        tql.save_q_table(q_table)
    else:
        q_table = tql.restore_q_table()

    if(q_table is not None): tql.q_testing(q_table, is_det)

    q_rewards, q_steps, q_expl_rates = dql.q_learning(is_det)
    dql.q_testing(is_det)

if __name__ == "__main__":
    main()
