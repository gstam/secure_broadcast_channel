import numpy as np
import matplotlib.pyplot as plt
from environment import Environment

def plot_reward_probability(lower_bound, upper_bound):
    for W_tx_Q1 in [True, False]:
        for W_tx_Q2 in [True, False]:
            reward_probability = []
            Pr_suc_rx_Q1_to_D1_list = []
            Pr_suc_rx_Q2_to_D2_list = []
            Pr_suc_rx_Q1_to_D2_list = []
            power_list = []
            for power1 in range(int(lower_bound+1), int(upper_bound)):
                power_list.append(power1)
                if W_tx_Q1 == True:
                    Pr_suc_rx_Q1_to_D1_list.append(env.get_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2))
                    Pr_suc_rx_Q1_to_D2_list.append(env.get_Pr_suc_rx_Q1_to_D2(power1, W_tx_Q2))
                else:
                    Pr_suc_rx_Q1_to_D1_list.append(.0)
                    Pr_suc_rx_Q1_to_D2_list.append(.0)
                
                if W_tx_Q2 == True:
                    Pr_suc_rx_Q2_to_D2_list.append(env.get_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1))
                else:
                    Pr_suc_rx_Q2_to_D2_list.append(.0)

                if W_tx_Q1 == True and W_tx_Q2 == True:
                    reward_probability.append(env.get_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2) * (1 - env.get_Pr_suc_rx_Q1_to_D2(power1, W_tx_Q2)) *env.get_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1))
                if W_tx_Q1 == True and W_tx_Q2 == False:
                    reward_probability.append(env.get_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2) * (1 - env.get_Pr_suc_rx_Q1_to_D2(power1, W_tx_Q2)))
                if W_tx_Q1 == False and W_tx_Q2 == True:
                    reward_probability.append(env.get_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1))
                if W_tx_Q1 == False and W_tx_Q2 == False:
                    reward_probability.append(.0)

            plt.plot(power_list, Pr_suc_rx_Q1_to_D1_list, label=f'Pr_suc_rx_Q1_to_D1')
            plt.plot(power_list, Pr_suc_rx_Q2_to_D2_list, label=f'Pr_suc_rx_Q2_to_D2')
            plt.plot(power_list, Pr_suc_rx_Q1_to_D2_list, label=f'Pr_suc_rx_Q1_to_D2')
            plt.plot(power_list, reward_probability, label=f'Reward Probability')
            plt.xlabel('Action (tx power)')
            plt.ylabel('Probability') 
            plt.legend()
            plt.savefig(f'W_tx_Q1_{W_tx_Q1}_W_tx_Q2_{W_tx_Q2}.jpg', format='jpg')
            plt.close()


if __name__ == '__main__':
    lambda_v = 1.
    Pr_arrival_Q1 = lambda_v
    B_threshold = 10 # queue capacity
    capacity_Q1 = B_threshold
    PathLoss_to_D1 = 2.2
    PathLoss_to_D2 = 2.2
    threshold1 = .3 #0.379433
    threshold2 = .2 #0.225893
    distance1 = 10 #8.2
    distance2 = 10 #14.6
    distance3 = 5
    power_max = 200 
    power_J = 50 #199.99
    g = 0.001 #0.008735
    q1 = 1. #0.8
    q2 = 1.
    P_max = 200

    episodes = 10
    episode_duration = 1000 # fix max_time because I don't get an error of exceeding the index in vectors describing the queue

    print_loss = False
    print_reward = False
    print_action = False
    
     
    env = Environment(capacity_Q1, Pr_arrival_Q1, lambda_v, PathLoss_to_D1, PathLoss_to_D2, threshold1, threshold2,  distance1, distance2, distance3, power_max, power_J, g, q1, q2, P_max)
    
    lower_bound = (threshold1 / (1 + threshold1))*P_max
    upper_bound = (1/(1 + threshold2))*P_max
    print(f'lower_bound: {lower_bound} upper_bound: {upper_bound}')

    plot_reward_probability(lower_bound, upper_bound)
    # for W_tx_Q1 in [True, False]:
    #     for W_tx_Q2 in [True, False]:
    #         Pr_suc_rx_Q1_to_D1_list = []
    #         Pr_suc_rx_Q2_to_D2_list = []
    #         Pr_suc_rx_Q1_to_D2_list = []
    #         power_list = []
    #         for power1 in range(int(lower_bound+1), int(upper_bound)):
    #             power_list.append(power1)
    #             if W_tx_Q1 == True:
    #                 Pr_suc_rx_Q1_to_D1_list.append(env.get_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2))
    #                 Pr_suc_rx_Q1_to_D2_list.append(env.get_Pr_suc_rx_Q1_to_D2(power1, W_tx_Q2))
    #             else:
    #                 Pr_suc_rx_Q1_to_D1_list.append(.0)
    #                 Pr_suc_rx_Q1_to_D2_list.append(.0)
                
    #             if W_tx_Q2 == True:
    #                 Pr_suc_rx_Q2_to_D2_list.append(env.get_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1))
    #             else:
    #                 Pr_suc_rx_Q2_to_D2_list.append(.0)
            
    #         plt.plot(power_list, Pr_suc_rx_Q1_to_D1_list, label=f'Pr_suc_rx_Q1_to_D1')
    #         plt.plot(power_list, Pr_suc_rx_Q2_to_D2_list, label=f'Pr_suc_rx_Q2_to_D2')
    #         plt.plot(power_list, Pr_suc_rx_Q1_to_D2_list, label=f'Pr_suc_rx_Q1_to_D2')
    #         plt.xlabel('Action (tx power)')
    #         plt.ylabel('Probability') 
    #         plt.legend()
    #         plt.savefig(f'W_tx_Q1_{W_tx_Q1}_W_tx_Q2_{W_tx_Q2}.jpg', format='jpg')
    #         plt.close()
    
            


