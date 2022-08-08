# TIN-TIN
# December 7th: I have only timeslots (not mini-slots). The agent performs an action at every timeslot.

# Reward function: function f

import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
import plot_actor_policy

                
class OUActionNoise:    
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self, num_states, num_actions, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            # target_actions = (lower_bound + 0.0001) + ((_target_actions + 1.)/2.)*(upper_bound - lower_bound)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
                )
            critic_value = critic_model(
                [state_batch, action_batch], training=True
                )
            # reduce_mean computes the mean of elements across dimensions of a tensor.
            critic_loss = tf.math.reduce_mean(
                tf.math.square(y - critic_value)
                )
        # critic_model.trainable_variables returns one value
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
            )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            # actions = (lower_bound + 0.0001) + ((_actions + 1.)/2.)*(upper_bound - lower_bound)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)
            
        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )
        # if print_loss == True:
        # print(f'\n --- critic_loss: {critic_loss} \t actor_loss:{actor_loss} --- \n')

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices for experiences
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
      
        self.update(state_batch, action_batch, reward_batch, next_state_batch)

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def get_actor(num_states):
    # Initialize weights
    last_init = tf.random_uniform_initializer(minval=-1.0, maxval=1.0) #(minval=-0.01, maxval=0.2)
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(16, activation="relu")(inputs) # 256
    out = layers.Dense(32, activation="relu")(out)    # 256
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out) #, kernel_initializer=last_init
    # # Upper bound 
    # gstam: I commented out the following line. It is not adequate to map outputs from the interval [-1, 1] to the interval [lower_bound, upper_bound]
    # gstam: I added code that does this mapping in the policy() function.
    # outputs = outputs * upper_bound 
    outputs = (lower_bound) + ((outputs + 1.0)/2.0)*(upper_bound - lower_bound)
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic(num_states, num_actions):
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input) #16
    state_out = layers.Dense(32, activation="relu")(state_out) # 16

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input) # 16

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(16, activation="relu")(concat) #256
    out = layers.Dense(32, activation="relu")(out) #256
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

def test_policy(actor_model, state, lower_bound, upper_bound):
    sampled_actions = tf.squeeze(actor_model(state))
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return np.squeeze(legal_action)

def policy(actor_model, state, noise_object, std, lower_bound, upper_bound, test_scenario):
    # actor_model(state) is a tensor of a value
    # rate_scalar = 0
    # while lambda_v >= rate_scalar: # and rate < 1:
    # gstam: 
    # map outputs from the interval [-1, 1] to the interval [0, 1] and subsequently to the interval [lower_bound, upper_bound]
    # the level of noise is also scaled up as a function of the [lower_bound, upper_bound] interval's length. This is a parameter of the DDPG algorithm that muste be carefully set.
    # clipping the sampled action to upper or lower bound MAY introduce a significant BIAS in the action selection process since the probability to select the upper and the lower bound are increased compared to the other actions. 
    # since proper measures have been taken to select sampled_actions in the interval [lower_bound, upper_bound] and ONLY noise can drop the sampled action
    # out of the target interval we just repeat random sampling 

    # std is epsilon in epsilon-greedy.
    # if np.random.default_rng().uniform(0., 1., 1) < std:
    #     # noise = np.random.normal(0, std)
    #     sampled_actions = np.random.uniform(lower_bound, upper_bound, 1)
    # else:
    #     _sampled_actions = tf.squeeze(actor_model(state))
    #     # noise = np.random.normal(0, std)
    #     # noise = noise_object()
    #     sampled_actions = (lower_bound + 0.0001) + ((_sampled_actions.numpy() + 1.)/2.)*(upper_bound - lower_bound)
    #     # noise = 0.0
    # std = 0.2
    # noise = np.random.normal(0, std)
    #sampled_actions = _sampled_actions + noise #(lower_bound + 0.0001) + ((_sampled_actions.numpy()+1.)/2.)*(upper_bound - lower_bound) + noise[0]*noise_scale_factor# Normalize selected action from [-1, 1] to [0, 1] and the scale up to [lower_bound, upper_bound
    _sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # noise = np.random.normal(0, std)
        # print(noise, _sampled_actions)
        #sampled_actions = (lower_bound + 0.0001) + ((_sampled_actions.numpy() + noise + 1.)/2.)*(upper_bound - lower_bound) # Normalize selected action from [-1, 1] to [0, 1] and the scale up to [lower_bound, upper_bound
    sampled_actions = _sampled_actions + noise
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    
    # legal_action = sampled_actions
    # print(f'State: {state} Sampled Action: {sa} Timeslot noise: {noise[0]} Scale Factor: {noise_scale_factor} Added Nosie: {noise[0]*noise_scale_factor}')

    # print(f"Sampled action: {sampled_actions}  Noise: {noise} Legal action: {legal_action}")
    # legal_action = sampled_actions      
    # We make sure action is within bounds
    # Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, 
    # values smaller than 0 become 0, and values larger than 1 become 1.
    #   sampled_actions = tf.squeeze(actor_model(state))
    #   noise = noise_object()
    #   sampled_actions = lower_bound + ((sampled_actions.numpy()+1.)/2.)*(upper_bound - lower_bound) + noise*(upper_bound - lower_bound)/10 # Normalize selected action from [-1, 1] to [0, 1] and the scale up to [lower_bound, upper_bound]
    #   legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return np.squeeze(legal_action)#sampled_actions #[np.squeeze(legal_action)]

def my_policy(state, lower_bound, upper_bound):
    backlog = state[0][0].numpy() 
    throughput = state[0][1].numpy()
    action = lower_bound
    # if backlog > 0: 
    #     action = upper_bound
    # else:
    #     action = lower_bound
    return action

def Q_value_plot(critic_model, capacity_Q1, lower_bound, upper_bound, episode):
    state_list = [s for s in range(capacity_Q1+1)]
    action_list = []
    for W_tx_Q1 in [True, False]:#, False]:
        for W_tx_Q2 in [True]: #, False]:
            for Q in range(capacity_Q1+1): #
                state = [float(W_tx_Q1), float(W_tx_Q2), float(Q)]
                tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                action_list = []
                critic_value_list = []
                for a in np.linspace(-1, 1, num=200):
                    raw_action = tf.squeeze(a) # range [-1, 1] actor_model(tf_state)
                    action = (lower_bound + 1e-06) + ((raw_action.numpy()+1.)/2.)*(upper_bound - lower_bound)
                    critic_value = critic_model([tf_state, tf.expand_dims(tf.convert_to_tensor(action), 0)], training=False)
                    action_list.append(action)
                    critic_value_list.append(critic_value.numpy()[0][0])

                # print(action_list)
                plt.plot(action_list, critic_value_list)
                plt.title(f'W_tx_Q1: {W_tx_Q1} \/ W_tx_Q2: {W_tx_Q2} \/ Backlog:{Q}')
                plt.xlabel('action')
                plt.ylabel('Qvalue')
                # plt.xlim([-1, 2])
                # plt.ylim([lower_bound, upper_bound])
                # plt.show()
                plt.savefig(f'W_tx_Q1_{W_tx_Q1}_W_tx_Q2_{W_tx_Q2}_Baclog_{Q}_Qvalue_{episode}.png')
                plt.close()

def plot_actor(actor_model, capacity_Q1, lower_bound, upper_bound):
    print(f'-----------------------------------------------------------')
    for W_tx_Q1 in [True, False]:
        for W_tx_Q2 in [True, False]:
            for Q in range(capacity_Q1+1):
                state = [float(W_tx_Q1), float(W_tx_Q2), float(Q)]
                tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                action = tf.squeeze(actor_model(tf_state))
                print(f'W_tx_Q1: {W_tx_Q1} W_tx_Q2: {W_tx_Q2} Q: {Q} action: {(lower_bound + 1e-06) + ((action.numpy()+1.)/2.)*(upper_bound - lower_bound)}')

#global PREVIOUS_PACKET_INDEX # this is the index of last packet in the previous timeslot
#global CURRENT_PACKET_INDEX # this is the index for the timeslot that just finished
#global CURRENT_TIME #this is the current running second in time

def define_parameters():
    lambda_v = 0.1
    Pr_arrival_Q1 = lambda_v
    B_threshold = 4 # queue capacity
    capacity_Q1 = B_threshold
    PathLoss_to_D1 = 2
    PathLoss_to_D2 = 2
    threshold1 = 0.5#0.5 #0.379433
    threshold2 = 0.4#0.4 #0.225893
    distance1 = 10 #8.2
    distance2 = 13 #14.6
    distance3 = 5
    power_max = 200 
    power_J = 20.0     #199.99
    g = 0.05         #0.008735
    q1 = 1.         #0.8
    q2 = 1.
    P_max = 200
    packet_rate_interval = 10
    return lambda_v, Pr_arrival_Q1, B_threshold, capacity_Q1, PathLoss_to_D1, PathLoss_to_D2, threshold1, threshold2, distance1,  distance2, distance3, power_max, power_J, g, q1, q2, P_max, packet_rate_interval


if __name__ == '__main__':
    test_scenario = False

    lambda_v, Pr_arrival_Q1, B_threshold, capacity_Q1, PathLoss_to_D1, PathLoss_to_D2, threshold1, threshold2, distance1,  distance2, distance3, power_max, power_J, g, q1, q2, P_max, packet_rate_interval = define_parameters()
    episodes = 300
    episode_duration = 1000 # fix max_time because I don't get an error of exceeding the index in vectors describing the queue
    env = Environment(capacity_Q1, Pr_arrival_Q1, lambda_v, PathLoss_to_D1, PathLoss_to_D2, threshold1, threshold2,  distance1, distance2, distance3, power_max, power_J, g, q1, q2, P_max, packet_rate_interval)

    lower_bound = (threshold1 / (1 + threshold1))*P_max
    upper_bound = (1/(1 + threshold2))*P_max
    print(f'Lower bound: {lower_bound} Upper bound: {upper_bound}')
    
    num_states = 2 # the state is the queue size
    num_actions = 1 # the action is the transmission power for packets from queue Q1 

    std_dev = 0.2*(upper_bound-lower_bound)#/1.0
    # std_dev_step = 0.1
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    actor_model = get_actor(num_states)
    critic_model = get_critic(num_states, num_actions)

    target_actor = get_actor(num_states)
    target_critic = get_critic(num_states, num_actions)

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Remove comment to load weights from previous runs.
    # if test_scenario == True:
    # actor_model.load_weights("tx_power_actor.h5")
    # critic_model.load_weights("tx_power_critic.h5")

    # target_actor.load_weights("tx_power_target_actor.h5")
    # target_critic.load_weights("tx_power_target_critic.h5")

    # Learning rate for actor-critic models
    critic_lr = 0.002#10**(-4) #0.002
    actor_lr = 0.001  #10**(-3) #0.001

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.005 #0.005 

    buffer = Buffer(num_states, num_actions, 50000, 64)
    
    log_file_name = 'training_logfile.csv'
    log_file = open(log_file_name, "w") #
    log_file.write(f'Episode;Timeslot;State;Action;Reward;Next_State\n')        
    
    for episode in range(1, episodes+1):
        total_episode_reward = 0
        state = env.reset()
        # print(f'Episode: {episode}')
        timeslot = 0
        
        # Reduce epsilon for the epsilon-greedy.
        if episode%5 == 0:
            print('This is a test scenario!')
            test_scenario = True
            figure_name = f'actor_policy_episode_{episode}.png'
            plot_actor_policy.plot_heatmap_actor_policy(actor_model, packet_rate_interval, capacity_Q1, lower_bound, upper_bound, figure_name)
        else: # and std_dev > 0.3:
            test_scenario = False
        #    std_dev -= std_dev_step

        while timeslot <= episode_duration:
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            
            if  test_scenario == True:
                action = test_policy(actor_model, tf_state, lower_bound, upper_bound)
            else:
                action = policy(actor_model, tf_state, ou_noise, std_dev, lower_bound, upper_bound, test_scenario)
            
            # action = my_policy(tf_state, lower_bound, upper_bound)
            reward, next_state = env.step(action)
            total_episode_reward += reward

            # Buffer management and learning
            if test_scenario == False:
                buffer.record((state, action, reward, next_state))
                buffer.learn()
                update_target(target_actor.variables, actor_model.variables, tau)
                update_target(target_critic.variables, critic_model.variables, tau)

            # Data logging
            log_file.write(f'{episode};{timeslot};{state};{action};{reward};{next_state}\n')        
            
            state = next_state
            timeslot += 1            
        
        print(f'Episode: {episode} \t Total Episode Reward: {total_episode_reward} Noise std: {std_dev}')
       
        # Save models after each episode.
        if test_scenario == False:
            actor_model.save_weights(f"tx_power_actor.h5")
            critic_model.save_weights(f"tx_power_critic.h5")
            target_actor.save_weights(f"tx_power_target_actor.h5")
            target_critic.save_weights(f"tx_power_target_critic.h5")    

    log_file.close()
    
    # files.download(log_file_name)
