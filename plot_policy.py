import rl_phy_sec
import matplotlib as plt
import tensorflow as tf

def policy_plot(actor_model, capacity_Q1, lower_bound, upper_bound):
    state_list = [s for s in range(capacity_Q1+1)]
    action_list = []
    for state in range(capacity_Q1+1):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        raw_action = tf.squeeze(actor_model(tf_state)) # range [-1, 1]
        action = (lower_bound + 1e-06) + ((raw_action.numpy()+1.)/2.)*(upper_bound - lower_bound)
        action_list.append(action)
    
    plt.bar(state_list, action_list)
    plt.xlabel('state')
    plt.ylabel('action')
    plt.xlim([-1, 2])
    plt.ylim([lower_bound, upper_bound])
    # plt.show()
    plt.savefig('policy.png')
    plt.close()


if __name__ == '__main__':
    
    lambda_v = 1.0
    Pr_arrival_Q1 = lambda_v
    B_threshold = 1 # queue capacity
    capacity_Q1 = B_threshold
    PathLoss_to_D1 =2.2
    PathLoss_to_D2 = 2.2
    threshold1 = 0.2 #0.379433
    threshold2 = 0.2 #0.225893
    distance1 = 10 #8.2
    distance2 = 10 #14.6
    distance3 = 5
    power_max = 200 
    power_J = 50 #199.99
    g = 0.1 #0.008735
    q1 = 1. #0.8
    q2 = 1.
    P_max = 200
    num_states = 1 # the state is the queue size
    num_actions = 1 # the action is the transmission power for packets from queue Q1 

    actor_model = rl_phy_sec.get_actor(num_states)
    critic_model = rl_phy_sec.get_critic(num_states, num_actions)

    target_actor = rl_phy_sec.get_actor(num_states)
    target_critic = rl_phy_sec.get_critic(num_states, num_actions)

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Remove comment to load weights from previous runs.
    actor_model.load_weights("./src/tx_power_actor.h5")
    critic_model.load_weights("./src/tx_power_critic.h5")

    target_actor.load_weights("./src/tx_power_target_actor.h5")
    target_critic.load_weights("./src/tx_power_target_critic.h5")
    
    lower_bound = (threshold1 / (1 + threshold1))*P_max
    upper_bound = (1/(1 + threshold2))*P_max
    
    policy_plot(actor_model, capacity_Q1, lower_bound, upper_bound)
