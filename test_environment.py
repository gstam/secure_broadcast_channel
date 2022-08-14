import numpy as np
import rl_phy_sec
import matplotlib.pyplot as plt
from environment import Environment

def plot_reward_probability(output_folder, env, lower_bound, upper_bound):
    for secrecy in [True, False]:
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
                        Pr_suc_rx_Q2_to_D2_list.append(env.get_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1, W_tx_Q2))
                    else:
                        Pr_suc_rx_Q2_to_D2_list.append(.0)

                    if W_tx_Q1 == True and W_tx_Q2 == True and secrecy == True:
                        reward_probability.append(env.get_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2) * env.get_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1, W_tx_Q2) * (1 - env.get_Pr_suc_rx_Q1_to_D2(power1, W_tx_Q2))) 
                    if W_tx_Q1 == True and W_tx_Q2 == False and secrecy == True:
                        reward_probability.append(env.get_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2) * (1 - env.get_Pr_suc_rx_Q1_to_D2(power1, W_tx_Q2)))
                    if W_tx_Q1 == True and W_tx_Q2 == True and secrecy == False:
                        reward_probability.append(env.get_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2) * env.get_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1, W_tx_Q2)) 
                    if W_tx_Q1 == True and W_tx_Q2 == False and secrecy == False:
                        reward_probability.append(env.get_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2))
                    if W_tx_Q1 == False and W_tx_Q2 == True:
                        reward_probability.append(env.get_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1, W_tx_Q2))
                    if W_tx_Q1 == False and W_tx_Q2 == False:
                        reward_probability.append(.0)

                plt.plot(power_list, Pr_suc_rx_Q1_to_D1_list, label=f'Pr_suc_rx_Q1_to_D1')
                plt.plot(power_list, Pr_suc_rx_Q2_to_D2_list, label=f'Pr_suc_rx_Q2_to_D2')
                plt.plot(power_list, Pr_suc_rx_Q1_to_D2_list, label=f'Pr_suc_rx_Q1_to_D2')
                plt.plot(power_list, reward_probability, label=f'Reward Probability')
                plt.xlabel('Action (tx power)')
                plt.ylabel('Probability') 
                plt.legend()
                plt.savefig(f'./Results/{output_folder}/te_W_tx_Q1_{W_tx_Q1}_W_tx_Q2_{W_tx_Q2}_Secrecy_{secrecy}.png', format='png')
                plt.close()

def plot_SD_reward_probability(output_folder, env, lower_bound, upper_bound):
    for secrecy in [True, False]:
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
                        Pr_suc_rx_Q1_to_D1_list.append(env.get_SD_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2))
                        Pr_suc_rx_Q1_to_D2_list.append(env.get_SD_Pr_suc_rx_Q1_to_D2(power1, W_tx_Q2))
                    else:
                        Pr_suc_rx_Q1_to_D1_list.append(0.0)
                        Pr_suc_rx_Q1_to_D2_list.append(0.0)
                    
                    if W_tx_Q2 == True:
                        Pr_suc_rx_Q2_to_D2_list.append(env.get_SD_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1, W_tx_Q2))
                    else:
                        Pr_suc_rx_Q2_to_D2_list.append(0.0)

                    if W_tx_Q1 == True and W_tx_Q2 == True and secrecy == True:
                        reward_probability.append(env.get_SD_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2) * env.get_SD_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1, W_tx_Q2) * (1 - env.get_SD_Pr_suc_rx_Q1_to_D2(power1, W_tx_Q2))) 
                    if W_tx_Q1 == True and W_tx_Q2 == False and secrecy == True:
                        reward_probability.append(env.get_SD_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2) * (1 - env.get_SD_Pr_suc_rx_Q1_to_D2(power1, W_tx_Q2)))
                    if W_tx_Q1 == True and W_tx_Q2 == True and secrecy == False:
                        reward_probability.append(env.get_SD_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2) * env.get_SD_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1, W_tx_Q2)) 
                    if W_tx_Q1 == True and W_tx_Q2 == False and secrecy == False:
                        reward_probability.append(env.get_SD_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2))
                    if W_tx_Q1 == False and W_tx_Q2 == True:
                        reward_probability.append(env.get_SD_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1, W_tx_Q2))
                    if W_tx_Q1 == False and W_tx_Q2 == False:
                        reward_probability.append(0.0)

                plt.plot(power_list, Pr_suc_rx_Q1_to_D1_list, label=f'Pr_suc_rx_Q1_to_D1')
                plt.plot(power_list, Pr_suc_rx_Q2_to_D2_list, label=f'Pr_suc_rx_Q2_to_D2')
                plt.plot(power_list, Pr_suc_rx_Q1_to_D2_list, label=f'Pr_suc_rx_Q1_to_D2')
                plt.plot(power_list, reward_probability, label=f'Joint Event Probability')
                plt.xlabel('Action (tx power)')
                plt.ylabel('Probability') 
                plt.legend()
                plt.savefig(f'./Results/{output_folder}/te_W_tx_Q1_{W_tx_Q1}_W_tx_Q2_{W_tx_Q2}_Secrecy_{secrecy}.png', format='png')
                plt.close()


def main(output_folder, successive_decoding):
    lambda_v, Pr_arrival_Q1, B_threshold, capacity_Q1, PathLoss_to_D1, PathLoss_to_D2, threshold1, threshold2, distance1,  distance2, distance3, power_max, power_J, g, q1, q2, P_max, packet_rate_interval, Q1_utilization_threshold, Q2_rate_threshold, TIN, SD = rl_phy_sec.define_parameters()
    env = Environment(capacity_Q1, Pr_arrival_Q1, lambda_v, PathLoss_to_D1, PathLoss_to_D2, threshold1, threshold2,  distance1, distance2, distance3, power_max, power_J, g, q1, q2, P_max, packet_rate_interval, Q1_utilization_threshold, Q2_rate_threshold, TIN, SD)
    
    lower_bound = (threshold1 / (1 + threshold1))*P_max
    upper_bound = (1/(1 + threshold2))*P_max
    print(f'lower_bound: {lower_bound} upper_bound: {upper_bound}')
    if successive_decoding == True:
        plot_SD_reward_probability(output_folder, env, lower_bound, upper_bound)
    else:
        plot_reward_probability(output_folder, env, lower_bound, upper_bound)

    return 0

if __name__ == '__main__':
    output_folder = '09_08_2022_15_08_02'
    successive_decoding = True
    main(output_folder, successive_decoding)
