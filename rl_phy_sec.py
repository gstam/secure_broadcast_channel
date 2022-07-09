# TIN-TIN
# December 7th: I have only timeslots (not mini-slots). The agent performs an action at every timeslot.

# Reward function: function f
# from google.colab import files

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import random
from  statistics import mean
from datetime import datetime

SUM_WEIGHTS = 0
lambda_v = 0.4
PathLoss = 2.2
threshold1 = 0.379433
threshold2 = 0.225893
distance1 = 8.2
distance2 = 14.6
distance3 = 10
power_max = 200 
power_J = 5 #199.99
g = 0.008735
B_threshold = 100 # queue capacity
q1 = 1 #0.8
q2 = 0.8
P_max = 200
timeslot_duration = 1 
delta_1 = 1 
delta_2 = -9  
d_ub = 5 

# Fix the global variables
CURRENT_TIME = 1
timeslot = 1 ################## I added this line
PREVIOUS_PACKET_INDEX = 0
CURRENT_PACKET_INDEX = 0

delay_less_than_dub_packet_counter = 0
delay_more_than_dub_packet_counter = 0

simulation_time_reduced = 100000 # change this also in SecureEnv class
# np.zeros array will have episodes number of element, indexing from 0 to episodes-1 
# mean_reward_per_episode = np.zeros(episodes)

packet_delay_threshold = 20 #40 #for TIN-TIN in seconds

episodes = 3
max_time = 1000 # fix max_time because I don't get an error of exceeding the index in vectors describing the queue
# Weights-> w_0: f, w_2: saturated queue throughput, w_3: size of queue
reward_weights = [1, 1, 1]
print_loss = False
print_reward = True
print_action = True
#def check_stability(lambda_v, rate):
#    if lambda_v > rate:
#        raise ValueError('Choose another transmission power')

def get_state_transition_reward(action, backlog_1, reward_weights, lambda_v, PathLoss, threshold1, threshold2, distance1, distance2, distance3, power_max, power_J, g, q1, q2):

    power1 = action
    power2 = power_max - power1

    successProb112 = np.exp((-(threshold1 * distance1**PathLoss)/(power1 - threshold1*power2))*(1 + power_J*g**2))*(1 - np.exp(-(threshold1 * distance2**PathLoss)/(power1 - threshold1*power2))*(1 + threshold1*(power_J/(power1 - threshold1*power2))*(distance2/distance3)**PathLoss)**(-1))                   
    successProb22 = np.exp(-(threshold2 * distance2**PathLoss)/power2)
    successProb11 = np.exp(((-threshold1* distance1**PathLoss)/power1)*(1 + power_J*g**2)) *(1-np.exp(((-threshold1*distance2**PathLoss)/power1))*(1 + threshold1*(power_J/power1)*(distance2/distance3)**PathLoss)**(-1))
    successProb212 = np.exp(-(threshold2 * distance2**PathLoss)/(power2 - threshold2*power1))*(1 + (threshold2 * power_J/(power2-threshold2*power1))*(distance2/distance3)**PathLoss)**(-1)

    theoretical_saturated_throughput =  successProb22 - (((1-q2*(1-q1))*successProb22 - q1*q2*successProb212)/(q1*q2*successProb112 + q1*(1-q2)*successProb11))*lambda_v
    #the service rate for the legitimate user
    rate = q1*q2*successProb112 + q1*(1-q2)*successProb11
    #print('Service rate (correct value): ', float(rate))
    if lambda_v <= rate:
        rate2 = theoretical_saturated_throughput
    else:
        rate2 =q2*(1-q1)*successProb22  + q2*q1*successProb212

    #---------------------------------- No secrecy -----------------------------------------------------#
    successProb112_no_secrecy = np.exp(-(threshold1 * distance1**PathLoss/(power1 - threshold1*power2)))
    successProb212_no_secrecy = np.exp(-(threshold2*distance2**PathLoss/(power2 - threshold2*power1)))
    successProb22_no_secrecy = np.exp(-(threshold2 * distance2**PathLoss)/power2)
    successProb11_no_secrecy = np.exp(-(threshold1 * distance1**PathLoss/power1))
    rate_no_secrecy = q1*q2*successProb112_no_secrecy + q1*(1-q2)*successProb11_no_secrecy

    f = rate/rate_no_secrecy
    
    reward = reward_weights[0]*f + reward_weights[1]*rate2 + reward_weights[2]*(1/(backlog_1 + 1))
    if print_reward == True:
      print(f'State: {backlog_1} \t Action: {action} \t f: {reward_weights[0]*f} \t Rate2: {reward_weights[1]*rate2} \t Backlog Reward: {reward_weights[2]*(1/(backlog_1 + 1))}')
    
    return reward, rate, rate2

def start_env(env,agent,action, lambda_v, simulation_time, timeslot_duration, delta_1, delta_2, d_ub):   
  #global PREVIOUS_PACKET_INDEX # this is the index of last packet in the previous timeslot
  #global CURRENT_PACKET_INDEX # this is the index for the timeslot that just finished
  global CURRENT_TIME #this is the current running second in time
  global timeslot # I added this line
  #global SUM_WEIGHTS
  
  # set the correct power level
  # power1 = action[0]
  # power2 = power_max - power1

  # successProb112 = np.exp((-(threshold1 * distance1**PathLoss)/(power1 - threshold1*power2))*(1 + power_J*g**2))*(1 - np.exp(-(threshold1 * distance2**PathLoss)/(power1 - threshold1*power2))*(1 + threshold1*(power_J/(power1 - threshold1*power2))*(distance2/distance3)**PathLoss)**(-1))                   
  # successProb22 = np.exp(-(threshold2 * distance2**PathLoss)/power2)
  # successProb11 = np.exp(((-threshold1* distance1**PathLoss)/power1)*(1 + power_J*g**2)) *(1-np.exp(((-threshold1*distance2**PathLoss)/power1))*(1 + threshold1*(power_J/power1)*(distance2/distance3)**PathLoss)**(-1))
  # successProb212 = np.exp(-(threshold2 * distance2**PathLoss)/(power2 - threshold2*power1))*(1 + (threshold2 * power_J/(power2-threshold2*power1))*(distance2/distance3)**PathLoss)**(-1)
  
  # theoretical_saturated_throughput =  successProb22 - (((1-q2*(1-q1))*successProb22 - q1*q2*successProb212)/(q1*q2*successProb112 + q1*(1-q2)*successProb11))*lambda_v
  # #the service rate for the legitimate user
  # rate = q1*q2*successProb112 + q1*(1-q2)*successProb11
  # #print('Service rate (correct value): ', float(rate))
  # rate2 = theoretical_saturated_throughput

  # #---------------------------------- No secrecy -----------------------------------------------------#
  # successProb112_no_secrecy = np.exp(-(threshold1*distance1**PathLoss/(power1 - threshold1*power2)))
  # successProb212_no_secrecy = np.exp(-(threshold2*distance2**PathLoss/(power2 - threshold2*power1)))
  # successProb22_no_secrecy = np.exp(-(threshold2 * distance2**PathLoss)/power2)
  # successProb11_no_secrecy = np.exp(-(threshold1 * distance1**PathLoss/power1))

  # rate_no_secrecy = q1*q2*successProb112_no_secrecy + q1*(1-q2)*successProb11_no_secrecy

  # function_f_Both_TIN = 1 - (rate/rate_no_secrecy)
  function_f_Both_TIN, rate, rate2 = get_state_transition_reward(action, env.queueSizeBur[CURRENT_TIME], reward_weights, lambda_v, PathLoss, threshold1, threshold2, distance1, distance2, distance3, power_max, power_J, g, q1, q2)

           
  """
  current_time : it is the duration of an episode
  """
  currentTime = CURRENT_TIME
  # bursty traffic 
  has_packet = env.queueSizeBur[currentTime] > 0
  has_arrival = random.random() < lambda_v
  has_departure = random.uniform(0,1) < rate
  has_space_left = env.queueSizeBur[currentTime] < B_threshold 
        
  #departures    
  if  has_packet: 
      if has_departure:
        # print('Packet departure')
        env.packet_departure(currentTime)
        env.received_Q1 += 1 
          
      else:
        # print('Packet retransmission') 
        env.trasmisson_delay[env.first_in_queue] = env.trasmisson_delay[env.first_in_queue] + 1            
  # arrivals
  env.received_Q2 = 0 ################# ---------------------- ???????????????????????????????????????
  if has_arrival and has_space_left: 
    # print('Packet arrival')
    env.packet_arrival(currentTime) 



  env.queueSizeBur[currentTime + 1] = env.queueSizeBur[currentTime]
  
  #------------------------------------------------------------------
  # saturated queue
  # arrivals
  env.queueSizeSat[currentTime] = env.packet_arrival_saturated(currentTime)
  has_packetSat = env.queueSizeSat[currentTime] > 0
  has_departureSat = random.uniform(0,1) < rate2
  # departures
  if has_packetSat:
      if has_departureSat:        
          env.packet_departure_saturated(rate2, currentTime)
          env.received_Q2 += 1

  #env.packet_nr = env.packet_nr - 1 # decrease the pointer
  saturated_throughput_reward = env.received_Q2       ############################/currentTime 
  transmission_delay_list =  env.trasmisson_delay[1:env.packet_nr] 
  #print('transmission_delay_list: ', transmission_delay_list)
  queuing_delay_list = env.total_delay[1:env.packet_nr]
  #print('queuing_delay_list: ', queuing_delay_list)

  # calculations only for assigning the weights
  weights = np.zeros(env.packet_nr)
  SUM_WEIGHTS = 0
  delay_reward_array = np.zeros(env.packet_nr)
  for packet in range(1,env.packet_nr):  
      if (queuing_delay_list[packet-1] + transmission_delay_list[packet-1]) <= d_ub:
          weights[packet-1] = delta_1
      else:
          weights[packet-1] = delta_2
  SUM_WEIGHTS = np.sum(weights)

  delay_reward = SUM_WEIGHTS/env.packet_nr
  #-------------------------------------------------------------

  average_packet_delay = np.sum(queuing_delay_list + transmission_delay_list)/env.packet_nr
  
  PREVIOUS_PACKET_INDEX = CURRENT_PACKET_INDEX
  CURRENT_TIME += timeslot_duration 

  return saturated_throughput_reward, delay_reward, rate, env.queueSizeBur[CURRENT_TIME], average_packet_delay, function_f_Both_TIN 

class Agent:
    """
    The state of the agent consists of the queue 
    We take into account all the characteristics of the queue such as length and delays
    """
    def __init__(self):
        pass
    
    def step(self, env, action):
        """
        This step method corresponds to the action for each episode
        -----
        It accepts the environment instance as an argument and allows the
        agent to perform the following actions:
        1) Observe the environment
        2) Make a decision about the action to take based on the observations
        3) Submit the action to the environment
        4) Get the reward for the current step
        """ 
        # the observation is the state of the environment
        current_obs = env.get_observation()
        #actions = env.get_actions()
        reward, rate, current_observation, average_packet_delay = env.action(action)
        return current_observation, reward, average_packet_delay

class Queue():
    def __init__(self, capacity):
        self.backlog = 0
        self.capacity = capacity

    def get_backlog(self):
        return self.backlog
    
    def get_capacity(self):
        return self.capacity
    
    def set_backlog(self, backlog):
        self.backlog = backlog
    
    def set_capacity(self, capacity):
        self.capacity = capacity
    
    def packet_arrival(self):
        if self.backlog < self.capacity:
            self.backlog = self.backlog + 1

    def packet_departure(self):
        self.backlog = self.backlog - 1
        
    def reset(self):
        self.backlog = 0

class BernoulliArrivalProcess():
    
    def init(self, success_probability):
        self.success_probability = success_probability
    
    def set_success_probability(self, success_probability):
        self.success_probability = success_probability
        
    def get_success_probability(self):
        return self.success_probability
    
    def generate_event(self):
        if np.random.default_rng().uniform(0., 1., 1) < self.success_probability:
            success = True
        else:
            success = False
        return success 


class Environment():
        
    def __init__(self, capacity, Pr_arrival_Q1):
        # Queue with Bernoulli arrivals and finite capacity.
        self.Q1 = Queue(capacity)
        self.Pr_arrival_Q1 = Pr_arrival_Q1
        
        # Saturated queue: It has a capacity of 1  but it never gets empty because we have a probability of 1 to get fresh arrival at each timeslot
        self.Q2 = Queue(capacity=1)
        self.Pr_arrival_Q2 = 1.

        # The environment is stochastic, so lots of probabilities...
        q1 = 1.0 #0.8
        q2 = 0.9
        self.Pr_tx_Q1 = q1
        self.Pr_tx_Q2 = q2
        self.lambda_v = 0.4
        self.PathLoss = 2.2
        self.threshold1 = 0.379433
        self.threshold2 = 0.225893
        self.distance1 = 8.2
        self.distance2 = 14.6
        self.distance3 = 10
        self.power_max = 200 
        self.power_J = 5 #199.99
        self.g = 0.008735
        self.B_threshold = 100 # queue capacity
        self.q1 = 1 #0.8
        self.q2 = 0.8
        self.P_max = 200
        self.timeslot_duration = 1 
        self.delta_1 = 1 
        self.delta_2 = -9  
        d_ub = 5 
        # The following probabilities should be functions of the environment's parameters.
        # self.Pr_suc_rx_Q1_to_D1 = Pr_suc_rx_Q1_to_D1
        # self.Pr_suc_rx_Q2_to_D2 = Pr_suc_rx_Q2_to_D2
        # self.Pr_suc_rx_Q1_to_D2 = Pr_suc_rx_Q1_to_D2 # Security constraint violation!
    
    # Probability that D1 will successfuly decode the packet sent by Q1. The computation depends on whether Q2 transmits or not.
    def get_Pr_suc_rx_Q1_to_D1(self, power1, W_tx_Q2):
        power2 = self.P_max - power1
        if W_tx_Q2 == True: # extra interference due to Q2's transmission
            Pr_suc_rx_Q1_to_D1 = np.exp((-(self.threshold1 * self.distance1**self.PathLoss)/(power1 - self.threshold1 * power2))*(1 + self.power_J*self.g**2)) 
        else: 
            Pr_suc_rx_Q1_to_D1 = np.exp(((-self.threshold1 * self.distance1**self.PathLoss)/power1)*(1 + self.power_J*self.g**2)) 
   
        return Pr_suc_rx_Q1_to_D1
    
    # Probability that D2 will successfuly decode the packet sent by Q2. The computation depends on whether Q1 transmits or not.
    def get_Pr_suc_rx_Q2_to_D2(self, power1, W_tx_Q1):
        power2 = self.P_max - power1
        if W_tx_Q1 == True:       # Jamming
            Pr_suc_rx_Q2_to_D2 =  np.exp(-(self.threshold2 * self.distance2**self.PathLoss)/(self.power2 - self.threshold2*self.power1))*(1 + (self.threshold2 * self.power_J/(power2-self.threshold2*power1))*(self.distance2/self.distance3)**self.PathLoss)**(-1)
        else:                           # No jamming
            Pr_suc_rx_Q2_to_D2 =  np.exp(-(self.threshold2 * self.distance2**self.PathLoss)/power2) 
        return Pr_suc_rx_Q2_to_D2

    # Probability that D2 will successfuly decode the packet sent by Q1. This is the secrecy violation scenario!
    def get_Pr_suc_rx_Q1_to_D2(self, power1, W_tx_Q2):
        power2 = self.P_max - power1
        if W_tx_Q2 == True:
            Pr_suc_rx_Q1_to_D2 =  np.exp(-(self.threshold1 * self.distance2**self.PathLoss)/(power1 - self.threshold1*power2))*(1 + self.threshold1*(self.power_J/(self.power1 - self.threshold1*self.power2))*(self.distance2/self.distance3)**self.PathLoss)**(-1)
        else:
            Pr_suc_rx_Q1_to_D2 =  np.exp(-(self.threshold1 * self.distance2**self.PathLoss)/(power1))*(1 + self.threshold1*(self.power_J/(self.power1))*(self.distance2/self.distance3)**self.PathLoss)**(-1)
        return Pr_suc_rx_Q1_to_D2

    def step(self, power1):
        # Get random transmissions from Q1 and Q2
        rnd = np.random.default_rng().uniform(0., 1., 1)
        if self.Q1.backlog > 0:
            if rnd < self.Pr_tx_Q1:
                W_tx_Q1 = True
            else:
                W_tx_Q1 = False
        
        rnd = np.random.default_rng().uniform(0., 1., 1)
        if rnd < self.Pr_tx_Q2:
            W_tx_Q2 = True
        else:
            W_tx_Q2 = False

        # Calculate probabilities to have a  successful reception at the destinations and the 
        # potential violation of the security constraint.
        if W_tx_Q1 == True:
            Pr_suc_rx_Q1_to_D1 = self.get_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2)
            Pr_suc_rx_Q1_to_D2 = self.get_Pr_suc_rx_Q1_to_D2(power1, W_tx_Q2)
        else:
            Pr_suc_rx_Q1_to_D1 = .0
        
        if W_tx_Q2 == True:
            Pr_suc_rx_Q2_to_D2 = self.get_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1)
        else:
            Pr_suc_rx_Q2_to_D2 = .0

        # Realization of the transmissions' outcomes and reward calculation
        rnd = np.random.default_rng().uniform(0., 1., 1)
        
        if rnd < Pr_suc_rx_Q1_to_D1:
            w_suc_rx_Q1_to_D1 = True
        else:
            w_suc_rx_Q1_to_D1 = False
        
        rnd = np.random.default_rng().uniform(0., 1., 1)
        if rnd < Pr_suc_rx_Q2_to_D2:
            w_suc_rx_Q2_to_D2 = True
        else:
            w_suc_rx_Q2_to_D2 = False
            
        rnd = np.random.default_rng().uniform(0., 1., 1)
        if rnd < Pr_suc_rx_Q1_to_D2:
            w_suc_rx_Q1_to_D2 = True
        else:
            w_suc_rx_Q1_to_D2 = False
        
        # Update state after successfull transmission.
        if w_suc_rx_Q1_to_D1:
            self.Q1.packet_departure()
        
        # Late arrivals at Q1
        rnd = np.random.default_rng().uniform(0., 1., 1)
        if rnd < self.Pr_arrival_Q1:
            self.Q1.packet_arrival()

        # Calculate Reward : Reward is provided only if 
        if W_tx_Q1 == True and W_tx_Q2 == True and w_suc_rx_Q1_to_D1 and w_suc_rx_Q2_to_D2 and (not w_suc_rx_Q1_to_D2):
            reward = 1
        if W_tx_Q1 == True and W_tx_Q2 == False and w_suc_rx_Q1_to_D1 and (not w_suc_rx_Q1_to_D2):
            reward = 1
        if W_tx_Q1 == False and W_tx_Q2 == True and w_suc_rx_Q2_to_D2:
            reward = 1
        else:
            reward = 0

        new_state = self.Q1.get_backlog()
        
        return reward, new_state

    def reset(self):
        self.Q1.set_backlog(0)
        return self.Q1.get_backlog()
                
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
    def __init__(self, buffer_capacity=100000, batch_size=64):
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
    # @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
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
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )
        if print_loss == True:
          print(f'critic_loss: {critic_loss} \t actor_loss:{actor_loss}')

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

def get_actor():
    # Initialize weights
    last_init = tf.random_uniform_initializer(minval=-1.0, maxval=1.0) #(minval=-0.01, maxval=0.2)
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs) 
    out = layers.Dense(256, activation="relu")(out)   
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    # # Upper bound 
    # gstam: I commented out the following line. It is not adequate to map outputs from the interval [-1, 1] to the interval [lower_bound, upper_bound]
    # gstam: I added code that does this mapping in the policy() function.
    # outputs = outputs * upper_bound 
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

def policy(state, noise_object, noise_scale_factor):
    # actor_model(state) is a tensor of a value
    # rate_scalar = 0
    # while lambda_v >= rate_scalar: # and rate < 1:
        # gstam: 
        # map outputs from the interval [-1, 1] to the interval [0, 1] and subsequently to the interval [lower_bound, upper_bound]
        # the level of noise is also scaled up as a function of the [lower_bound, upper_bound] interval's length. This is a parameter of the DDPG algorithm that muste be carefully set.
        # clipping the sampled action to upper or lower bound MAY introduce a significant BIAS in the action selection process since the probability to select the upper and the lower bound are increased compared to the other actions. 
        # since proper measures have been taken to select sampled_actions in the interval [lower_bound, upper_bound] and ONLY noise can drop the sampled action
        # out of the target interval we just repeat random sampling 
    denominator = 1
    is_legal_action = False
    while not is_legal_action:
        _sampled_actions = tf.squeeze(actor_model(state))
        # sa = sampled_actions
        noise = noise_object()
        sampled_actions = (lower_bound + 0.0001) + ((_sampled_actions.numpy()+1.)/2.)*(upper_bound - lower_bound) + noise[0]*noise_scale_factor# Normalize selected action from [-1, 1] to [0, 1] and the scale up to [lower_bound, upper_bound
        # print(f'State: {state} Sampled Action: {sa} Timeslot noise: {noise[0]} Scale Factor: {noise_scale_factor} Added Nosie: {noise[0]*noise_scale_factor}')
        if sampled_actions >= lower_bound and sampled_actions <= upper_bound:
          if print_action == True:
            print(f'Model action: {(lower_bound + 0.0001) + (( _sampled_actions.numpy()+1.)/2.)*(upper_bound - lower_bound)} Action with noise: {sampled_actions}')
          is_legal_action = True
        
        # legal_action = sampled_actions      
        # We make sure action is within bounds
        # Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, 
        # values smaller than 0 become 0, and values larger than 1 become 1.
        #   sampled_actions = tf.squeeze(actor_model(state))
        #   noise = noise_object()
        #   sampled_actions = lower_bound + ((sampled_actions.numpy()+1.)/2.)*(upper_bound - lower_bound) + noise*(upper_bound - lower_bound)/10 # Normalize selected action from [-1, 1] to [0, 1] and the scale up to [lower_bound, upper_bound]
        #   legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
      
    # gstam: 
    # print(f'Lower Bound: {lower_bound} Upper Bound: {upper_bound} Noise: {noise} Sampled_actions: {sampled_actions} legal_action: {legal_action}')
    # power1 = legal_action
    # power2 = P_max - power1
    # successProb112 = np.exp((-(threshold1 * distance1**PathLoss)/(power1 - threshold1*power2))*(1 + power_J*g**2))*(1 - np.exp(-(threshold1 * distance2**PathLoss)/(power1 - threshold1*power2))*(1 + threshold1*(power_J/(power1 - threshold1*power2))*(distance2/distance3)**PathLoss)**(-1))                   
    # successProb11 = np.exp(((-threshold1* distance1**PathLoss)/power1)*(1 + power_J*g**2)) *(1-np.exp(((-threshold1*distance2**PathLoss)/power1))*(1 + threshold1*(power_J/power1)*(distance2/distance3)**PathLoss)**(-1))
    # rate = q1*q2*successProb112 + q1*(1-q2)*successProb11
    # rate_scalar = tf.reshape([rate], []).numpy()
    # print('Legal_action: ', legal_action)
    return sampled_actions #[np.squeeze(legal_action)]

#global PREVIOUS_PACKET_INDEX # this is the index of last packet in the previous timeslot
#global CURRENT_PACKET_INDEX # this is the index for the timeslot that just finished
#global CURRENT_TIME #this is the current running second in time

def policy_plot(actor_model):
    state_list = [s for s in range(B_threshold)]
    action_list = []
    for state in range(B_threshold):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        raw_action = tf.squeeze(actor_model(tf_state)) # range [-1, 1]
        action = (lower_bound + 1e-06) + ((raw_action.numpy()+1.)/2.)*(upper_bound - lower_bound)
        action_list.append(action)
    
    plt.scatter(state_list, action_list)
    plt.xlabel('state')
    plt.ylabel('action')
    plt.ylim([55, 164])
    plt.show()
    

if __name__ == '__main__':
    env = Environment()
    agent = Agent()

    lower_bound = (threshold1 / (1 + threshold1))*P_max
    upper_bound = (1/(1 + threshold2))*P_max

    num_states = 1 # the state is the queue size
    num_actions = 1 # the action is the transmission power for packets from queue Q1 

    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    actor_model = get_actor()
    critic_model = get_critic()

    target_actor = get_actor()
    target_critic = get_critic()
   

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Remove comment to load weights from previous runs.
    # actor_model.load_weights("tx_power_actor.h5")
    # critic_model.load_weights("tx_power_critic.h5")

    # target_actor.load_weights("tx_power_target_actor.h5")
    # target_critic.load_weights("tx_power_target_critic.h5")

    # Learning rate for actor-critic models
    critic_lr = 0.002 #10**(-4) #0.002
    actor_lr = 0.001  #10**(-3) #0.001

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.005 

    buffer = Buffer(50000, 64)
    
    log_file_name = 'training_logfile.csv'
    log_file = open(log_file_name, "w") #
    log_file.write(f'Episode;Timeslot;State;Action;Reward;Next_State;Average Packet Delay\n')        
    noise_denominator = 1.
    noise_scale_factor = (upper_bound - lower_bound)/noise_denominator  
    
    for episode in range(1, episodes+1):
        state = env.reset()
        CURRENT_TIME = 1
        # print(f'Episode: {episode}')
        timeslot = 0         

        if episode % 1 == 0 and noise_denominator <= 128:
            noise_denominator *= 2
            noise_scale_factor = (upper_bound - lower_bound)/noise_denominator  
            print(f'Noise_denominator: {noise_denominator}')
        
        while True:  #average_packet_delay < packet_delay_threshold:
            # Env step based on action
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            # Choose action given a policy.
            action = policy(tf_state, ou_noise, noise_scale_factor)
            next_state, timeslot_reward, average_packet_delay = agent.step(env, action)
            total_episode_reward += timeslot_reward

            # Buffer management and learning
            if timeslot % 200 == 0:
              print_action = True
              print_loss = False
              print_reward = False
            else:
              print_action = False
              print_loss = False
              print_reward = False

            buffer.record((state, action, timeslot_reward, next_state))
            buffer.learn()
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            # Data logging
            log_file.write(f'{episode};{timeslot};{state};{action};{timeslot_reward};{next_state};{average_packet_delay}\n')        
            
            state = next_state
            
            # if CURRENT_TIME % 1000 == 0:
            #     print(f'Timeslot: {CURRENT_TIME} \t Accumulated Episode Reward: {total_episode_reward}')
            # Check if end of episode        
            if timeslot >= max_time:
                # print('End of episode:', CURRENT_TIME)
                print(f'Episode: {episode} \t Total Episode Reward: {total_episode_reward}')
                policy_plot(actor_model)  
                break
            
            timeslot += 1
        
    log_file.close()

    actor_model.save_weights("tx_power_actor.h5")
    critic_model.save_weights("tx_power_critic.h5")

    target_actor.save_weights("tx_power_target_actor.h5")
    target_critic.save_weights("tx_power_target_critic.h5")

    # files.download(log_file_name)