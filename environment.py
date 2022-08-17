import numpy as np
from fifo_queue import Queue

class Environment():

    def __init__(self, capacity, Pr_arrival_Q1, lambda_v, PathLoss_to_D1, PathLoss_to_D2, threshold1, threshold2, 
                    distance1, distance2, distance3, power_max, power_J, g, q1, q2, P_max, packet_tx_rate_interval,
                    Q1_utilization_threshold, Q2_rate_threshold, successive_decoding):
        self.timer = 1
        self.packet_tx_rate_interval = packet_tx_rate_interval
        self.successive_decoding = successive_decoding
        # Queue with Bernoulli arrivals and finite capacity.
        self.Q1 = Queue(capacity)
        self.Pr_arrival_Q1 = Pr_arrival_Q1
        
        # Saturated queue: It has a capacity of 1  but it never gets empty because we have a probability of 1 to get fresh arrival at each timeslot
        self.Q2 = Queue(capacity=1)
        self.Pr_arrival_Q2 = 1.
        self.N = 0
        self.packet_rate_Q2 = self._Q2_packet_rate()
        self.Pr_tx_Q1 = q1
        self.Pr_tx_Q2 = q2
        self.lambda_v = lambda_v
        self.PathLoss_to_D1 = PathLoss_to_D1
        self.PathLoss_to_D2 = PathLoss_to_D2
        self.threshold1 = threshold1
        self.threshold2 =threshold2
        self.distance1 = distance1
        self.distance2 = distance2
        self.distance3 = distance3
        self.power_max = power_max 
        self.power_J = power_J #199.99
        self.g = g
        self.q1 = q1 #0.8
        self.q2 = q2
        self.P_max = P_max

        self.lower_bound = (self.threshold1 / (1 + self.threshold1))*self.P_max
        self.upper_bound = (1/(1 + self.threshold2))*self.P_max

        self.Q1_utilization_threshold = Q1_utilization_threshold
        self.Q2_rate_threshold = Q2_rate_threshold
        self.scheduled_transmissions = [0, 1]

        # Statistics
        self.reward_interval = 100
        self.count_packets_Q2_to_D2 = 0
        self.Q1_tx_packets = 0
        self.Q1_packets_with_secrecy = 0

    def _Q2_packet_rate(self):
        return  self.N/self.timer #self.timer self.rate_interval


        # The following probabilities should be functions of the environment's parameters.
        # self.Pr_suc_rx_Q1_to_D1 = Pr_suc_rx_Q1_to_D1
        # self.Pr_suc_rx_Q2_to_D2 = Pr_suc_rx_Q2_to_D2
        # self.Pr_suc_rx_Q1_to_D2 = Pr_suc_rx_Q1_to_D2 # Security constraint violation!
    def _calculate_Q2_tx_power(self, power, W_tx_Q2):
        if W_tx_Q2 == False:
            power_2 = 0
        else:
            power_2 = self.P_max - power
        return power_2 #np.maximum(np.minimum( (self.P_max - power), self.upper_bound ), self.lower_bound)


    # Probability that D1 will successfuly decode the packet sent by Q1. The computation depends on whether Q2 transmits or not.
    def get_Pr_suc_rx_Q1_to_D1(self, power1, W_tx_Q2, successive_decoding):
        power2 = self._calculate_Q2_tx_power(power1, W_tx_Q2)
        if W_tx_Q2 == False:
             Pr_suc_rx_Q1_to_D1 = np.exp(((-self.threshold1 * self.distance1**self.PathLoss_to_D1)/power1) *(1 + self.power_J*self.g**2))
        elif W_tx_Q2 == True and successive_decoding == False:
            if power1 > self.threshold1*power2:
                Pr_suc_rx_Q1_to_D1 = np.exp(-(self.threshold1 * self.distance1**self.PathLoss_to_D1) * (1 + self.power_J*self.g**2)/(power1 - self.threshold1 * power2))
            else:
                Pr_suc_rx_Q1_to_D1 = 0.0
        elif W_tx_Q2 == True and successive_decoding == True:
            if power2 > self.threshold2*power1:
                part_A = (self.threshold2*(1 + self.g**2 * self.power_J)*self.distance1**self.PathLoss_to_D1)/(power2 - self.threshold2*power1)
            else:
                part_A = 0.0
            part_B = self.threshold1*(1 + self.g**2 * self.power_J)*self.distance1**self.PathLoss_to_D1/power1
            max_part = np.max(part_A, part_B)
            Pr_suc_rx_Q1_to_D1 = np.exp(-max_part)
        else:
            Pr_suc_rx_Q1_to_D1 = 0.0
        
        return Pr_suc_rx_Q1_to_D1


    # Probability that D2 will successfuly decode the packet sent by Q2. The computation depends on whether Q1 transmits or not.
    def get_Pr_suc_rx_Q2_to_D2(self, power1, W_tx_Q1,  W_tx_Q2):
        power2 = self._calculate_Q2_tx_power(power1, W_tx_Q2)
        if W_tx_Q1 == False:
            Pr_suc_rx_Q2_to_D2 =  np.exp(-(self.threshold2 * self.distance2**self.PathLoss_to_D2)/power2) 
        else: # W_tx_Q1 == True:
            if power2 > self.threshold2*power1:
                Pr_suc_rx_Q2_to_D2 =  np.exp(-(self.threshold2 * self.distance2**self.PathLoss_to_D2)/(power2 - self.threshold2*power1))/(1 + (self.threshold2 * self.power_J/(power2-self.threshold2*power1))*((self.distance2/self.distance3)**self.PathLoss_to_D2))
            else:
                Pr_suc_rx_Q2_to_D2 = 0.0
        return Pr_suc_rx_Q2_to_D2


    # Probability that D2 will successfuly decode the packet sent by Q1. This is the secrecy violation scenario!
    def get_Pr_suc_rx_Q1_to_D2(self, power1, W_tx_Q2):
        power2 = self._calculate_Q2_tx_power(power1, W_tx_Q2)
        if W_tx_Q2 == False:
            Pr_suc_rx_Q1_to_D2 = np.exp(-self.threshold1 * self.distance2**self.PathLoss_to_D2/power1)/(1 + self.threshold1*self.power_J * ((self.distance2/self.distance3)**self.PathLoss_to_D2)/power1)
        else:
            if power1 > self.threshold1*power2:
                Pr_suc_rx_Q1_to_D2 = np.exp(-(self.threshold1 * self.distance2**self.PathLoss_to_D2)/(power1 - self.threshold1*power2))/(1 + (self.threshold1*self.power_J)*(self.distance2/self.distance3)**self.PathLoss_to_D2 / (power1 - self.threshold1*power2))
            else:
                Pr_suc_rx_Q1_to_D2 = 0.0
    
        return Pr_suc_rx_Q1_to_D2


    def compute_reward(self, power1, W_tx_Q1, W_tx_Q2, w_suc_rx_Q1_to_D1, w_suc_rx_Q2_to_D2, w_suc_rx_Q1_to_D2):
        reward = .0
        if W_tx_Q1 == True and W_tx_Q2 == True:
            if w_suc_rx_Q1_to_D1 and w_suc_rx_Q2_to_D2:
                reward += 2
                # if (not w_suc_rx_Q1_to_D2):
                #     reward += 10
            elif w_suc_rx_Q1_to_D1 and not w_suc_rx_Q2_to_D2:
                reward += 1
                # if (not w_suc_rx_Q1_to_D2):
                #     reward += 10
            elif not w_suc_rx_Q1_to_D1 and w_suc_rx_Q2_to_D2:
                reward += 1
            else:
                reward += 0.0
        elif W_tx_Q1 == True and W_tx_Q2 == False and w_suc_rx_Q1_to_D1: # and (not w_suc_rx_Q1_to_D2):
            reward += 1.
            # if (not w_suc_rx_Q1_to_D2):
            #     reward += 10
        elif W_tx_Q1 == False and W_tx_Q2 == True and w_suc_rx_Q2_to_D2:
            reward += 0.
        else:
            reward = 0
        
        # Reinforce empty the action resulted in an empyt Q1
        # if (self.Q1.backlog - w_suc_rx_Q1_to_D1) == 0:
        #     reward += 5
        
        # # Reinforce the secrecy constraint.
        # if W_tx_Q1 == True and not w_suc_rx_Q1_to_D2:
        #     reward += 1
        return reward


    def compute_transition_reward(self, next_state, w_suc_rx_Q1_to_D1, w_suc_rx_Q1_to_D2):
        secrecy_reward = .0
        # if w_suc_rx_Q1_to_D2 == False and w_suc_rx_Q1_to_D1 == True:
        #     secrecy_reward += 10.0
        #     self.Q1_packets_with_secrecy += 1
        
        Q1_utilization = next_state[0]
        Q2_running_packet_rate = next_state[1]

        reward = 0.0
        if Q1_utilization <= self.Q1_utilization_threshold and Q2_running_packet_rate > self.Q2_rate_threshold:
            reward = 10.0*Q2_running_packet_rate 
        
        # The following reward functions don't work really well. 
            # if next_state[0] <= 0.5:
            #     reward = 10*next_state[1] #next_state[1] is calculated using rate_interval
                
            #w_1*10*(1 - float(self.Q1.backlog)/self.Q1.capacity) + (1-w_1)*10*float(self.packet_rate_Q2)  + secrecy_reward
        return reward + secrecy_reward  
        #return (1-w_1)*float(self.packet_rate_Q2)


    def get_tx_success_probabilities(self, W_tx_Q1, W_tx_Q2, power1, successive_decoding):
         # Calculate probabilities to have a successful reception at the destinations and the 
        # potential violation of the security constraint.
        Pr_suc_rx_Q1_to_D1 = .0
        Pr_suc_rx_Q1_to_D2 = .0
        Pr_suc_rx_Q2_to_D2 = .0

        if W_tx_Q1 == True:
            Pr_suc_rx_Q1_to_D1 = self.get_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2, successive_decoding)
            Pr_suc_rx_Q1_to_D2 = self.get_Pr_suc_rx_Q1_to_D2(power1, W_tx_Q2)
        
        if W_tx_Q2 == True:
            Pr_suc_rx_Q2_to_D2 = self.get_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1, W_tx_Q2)
        
        return Pr_suc_rx_Q1_to_D1, Pr_suc_rx_Q1_to_D2, Pr_suc_rx_Q2_to_D2


    def step(self, power1):
        W_tx_Q1 = self.scheduled_transmissions[0]
        W_tx_Q2 = self.scheduled_transmissions[1]

        Pr_suc_rx_Q1_to_D1, Pr_suc_rx_Q1_to_D2, Pr_suc_rx_Q2_to_D2 = self.get_tx_success_probabilities(W_tx_Q1, W_tx_Q2, power1, self.successive_decoding)

        # Realization of the transmissions' outcomes and reward calculation
        rnd = np.random.default_rng().uniform(0., 1., 1)
        w_suc_rx_Q1_to_D1 = False
        if rnd < Pr_suc_rx_Q1_to_D1:
            w_suc_rx_Q1_to_D1 = True
            
        w_suc_rx_Q2_to_D2 = False
        rnd = np.random.default_rng().uniform(0., 1., 1)
        if rnd < Pr_suc_rx_Q2_to_D2:
            w_suc_rx_Q2_to_D2 = True

        w_suc_rx_Q1_to_D2 = False            
        rnd = np.random.default_rng().uniform(0., 1., 1)
        if rnd < Pr_suc_rx_Q1_to_D2:
            w_suc_rx_Q1_to_D2 = True

        # Next State.
        if w_suc_rx_Q1_to_D1:
            self.Q1.packet_departure()
        
        # Late arrivals at Q1
        rnd = np.random.default_rng().uniform(0., 1., 1)
        if rnd < self.Pr_arrival_Q1:
            self.Q1.packet_arrival()

        # Q2 throughput update
        if w_suc_rx_Q2_to_D2 == True:
            self.N += 1
        self.packet_rate_Q2 = self._Q2_packet_rate() #self.N/self.timer #self.rate_interval# self.timer 

        # Will Q1 transmit in the next timeslot?
        rnd = np.random.default_rng().uniform(0., 1., 1)
        self.scheduled_transmissions[0] = False
        if self.Q1.backlog > 0 and rnd < self.Pr_tx_Q1:
            self.scheduled_transmissions[0] = True
    
        # Will Q2 transmit in the next timeslot?
        rnd = np.random.default_rng().uniform(0., 1., 1)
        self.scheduled_transmissions[1] = False
        if rnd < self.Pr_tx_Q2:
            self.scheduled_transmissions[1] = True
    
        #float(self.scheduled_transmissions[0]), float(self.scheduled_transmissions[1]),
        new_state = [float(self.Q1.backlog/self.Q1.capacity), float(self.packet_rate_Q2)]

        # Compute reward
        #reward = self.compute_reward(power1, W_tx_Q1, W_tx_Q2, w_suc_rx_Q1_to_D1, w_suc_rx_Q2_to_D2, w_suc_rx_Q1_to_D2)
        reward = self.compute_transition_reward(new_state, w_suc_rx_Q1_to_D1, w_suc_rx_Q1_to_D2)

        if self.timer > self.packet_tx_rate_interval:
            self.timer = 1
            self.N = 0
        else:
            self.timer += 1
        return reward, new_state

    def reset(self):
        self.timer = 1
        self.scheduled_transmissions = [0, 0]
        self.Q1.set_backlog(0)
        self.N = 0
        self.Q1_packets_with_secrecy = 0
        self.packet_rate_Q2 = self._Q2_packet_rate() #self.N/self.timer#self.rate_interval #self.timer 
        return  [float(self.Q1.backlog/self.Q1.capacity), float(self.packet_rate_Q2)]

    # # Probability that D1 will successfuly decode the packet sent by Q1. The computation depends on whether Q2 transmits or not.
    # def get_SD_Pr_suc_rx_Q1_to_D1(self, power1, W_tx_Q2):
    #     power2 = self._calculate_Q2_tx_power(power1, W_tx_Q2)
    #     # Equation (7) first part (before the multiplication symbol X)
    #     if W_tx_Q2 == True:  
    #         if self.threshold2 * power1 < power2 and power2 <= power1 * self.threshold2*(1 + self.threshold1)/self.threshold1:
    #             Pr_suc_rx_Q1_to_D1 = np.exp(-(self.threshold2 * self.distance1**self.PathLoss_to_D1)/(power2 - self.threshold2 * power1))# * (1 + self.power_J*self.g**2))
    #         elif  power2 > power1 * self.threshold2 * (1 + self.threshold1)/self.threshold1:
    #             Pr_suc_rx_Q1_to_D1 = np.exp(((-self.threshold1 * self.distance1**self.PathLoss_to_D1)/power1))
    #         else:
    #             Pr_suc_rx_Q1_to_D1 = 0.0 #np.exp(((-self.threshold1 * self.distance1**self.PathLoss_to_D1)/power1))
    #             # print('Error 1: get_SD_Pr_suc_rx_Q1_to_D1, The conditions provided in the paper are not exclusive?')
    #     else:
    #         if power1 > self.threshold1*power2:
    #             Pr_suc_rx_Q1_to_D1 = np.exp(((-self.threshold1 * self.distance1**self.PathLoss_to_D1)/power1))#*(1 + self.power_J*self.g**2))#no jamming.
    #         else:
    #             Pr_suc_rx_Q1_to_D1 = 0.0

    #     return Pr_suc_rx_Q1_to_D1

        
    # # Probability that D2 will successfuly decode the packet sent by Q2. The computation depends on whether Q1 transmits or not.
    # def get_SD_Pr_suc_rx_Q2_to_D2(self, power1, W_tx_Q1,  W_tx_Q2):
    #     power2 = self._calculate_Q2_tx_power(power1, W_tx_Q2)
    #     if W_tx_Q1 == True:
    #         if power2 > self.threshold2*power1:
    #             Pr_suc_rx_Q2_to_D2 =  np.exp(-(self.threshold2 * self.distance2**self.PathLoss_to_D2)/(power2 - self.threshold2*power1))#/(1 + (self.threshold2 * self.power_J/(power2-self.threshold2*power1))*((self.distance2/self.distance3)**self.PathLoss_to_D2))
    #         else:
    #             Pr_suc_rx_Q2_to_D2 = 0.0
    #     else:
    #         if power2 > self.threshold2*power1:                           # No jamming
    #             Pr_suc_rx_Q2_to_D2 =  np.exp(-(self.threshold2 * self.distance2**self.PathLoss_to_D2)/power2) 
    #         else:
    #             Pr_suc_rx_Q2_to_D2 = 0.0
    #     return Pr_suc_rx_Q2_to_D2

    # # Probability that D2 will successfuly decode the packet sent by Q1. This is the secrecy violation scenario!
    # def get_SD_Pr_suc_rx_Q1_to_D2(self, power1, W_tx_Q2):
    #     power2 = self._calculate_Q2_tx_power(power1, W_tx_Q2)
    #     if W_tx_Q2 == True:
    #         if power1 > self.threshold1*power2:
    #             Pr_suc_rx_Q1_to_D2 =  np.exp(-(self.threshold1 * self.distance2**self.PathLoss_to_D2)/(power1 - self.threshold1*power2)) #/(1 + (self.threshold1*self.power_J)/(power1 - self.threshold1*power1)*(self.distance2/self.distance3)**self.PathLoss_to_D2)
    #         else:
    #             Pr_suc_rx_Q1_to_D2 = 0.0
    #     else:
    #         if power1 > self.threshold1*power2:
    #             Pr_suc_rx_Q1_to_D2 =  np.exp(-self.threshold1 * self.distance2**self.PathLoss_to_D2/power1) #/(1 + self.threshold2*self.power_J/power1*((self.distance2/self.distance3)**self.PathLoss_to_D2))
    #         else:
    #             Pr_suc_rx_Q1_to_D2 = 0.0
    #     return Pr_suc_rx_Q1_to_D2
