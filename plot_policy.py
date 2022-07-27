import numpy as np
import rl_phy_sec
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

def policy_plot(actor_model, capacity_Q1, lower_bound, upper_bound):
    state_list = [s for s in range(capacity_Q1+1)]
    action_list = []
    for W_tx_Q1 in [True, False]:
        for W_tx_Q2 in [True, False]:
            action_list = []
            for Q in range(capacity_Q1+1):
                state = [float(W_tx_Q1), float(W_tx_Q2), float(Q)]
                tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                raw_action = tf.squeeze(actor_model(tf_state)) # range [-1, 1]
                action = (lower_bound + 1e-06) + ((raw_action.numpy()+1.)/2.)*(upper_bound - lower_bound)
                action_list.append(action)
            print(action_list)
            plt.bar(state_list, action_list)
            plt.title(f'W_tx_Q1: {W_tx_Q1} W_tx_Q2: {W_tx_Q2}')
            plt.xlabel('state')
            plt.ylabel('action')
            # plt.xlim([-1, 2])
            # plt.ylim([lower_bound, upper_bound])
            # plt.show()
            plt.savefig(f'pp_W_tx_Q1_{W_tx_Q1}_W_tx_Q2_{W_tx_Q2}_policy.png')
            plt.close()




def main():
    lambda_v, Pr_arrival_Q1, B_threshold, capacity_Q1, PathLoss_to_D1, PathLoss_to_D2, threshold1, threshold2, distance1,  distance2, distance3, power_max, power_J, g, q1, q2, P_max = rl_phy_sec.define_parameters()
    num_states = 3 # the state is the queue size
    num_actions = 1 # the action is the transmission power for packets from queue Q1 

    actor_model = rl_phy_sec.get_actor(num_states)
    critic_model = rl_phy_sec.get_critic(num_states, num_actions)

    target_actor = rl_phy_sec.get_actor(num_states)
    target_critic = rl_phy_sec.get_critic(num_states, num_actions)

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Remove comment to load weights from previous runs.
    actor_model.load_weights("tx_power_actor.h5")
    critic_model.load_weights("tx_power_critic.h5")

    target_actor.load_weights("tx_power_target_actor.h5")
    target_critic.load_weights("tx_power_target_critic.h5")
    
    lower_bound = (threshold1 / (1 + threshold1))*P_max
    upper_bound = (1/(1 + threshold2))*P_max
    
  
    policy_plot(target_actor, capacity_Q1, lower_bound, upper_bound)
    # Q_value_plot(target_actor, critic_model, capacity_Q1, lower_bound, upper_bound)
    return 0

if __name__ == '__main__':
    main()
  
    
    
