import numpy as np
from fifo_queue import Queue

class Environment():

    def __init__(self, capacity, Pr_arrival_Q1, lambda_v, PathLoss_to_D1, PathLoss_to_D2, threshold1, threshold2, 
                    distance1, distance2, distance3, power_max, power_J, g, q1, q2, P_max):
        self.timer = 1
        self.rate_interval = 4
        # Queue with Bernoulli arrivals and finite capacity.
        self.Q1 = Queue(capacity)
        self.Pr_arrival_Q1 = Pr_arrival_Q1
        
        # Saturated queue: It has a capacity of 1  but it never gets empty because we have a probability of 1 to get fresh arrival at each timeslot
        self.Q2 = Queue(capacity=1)
        self.Pr_arrival_Q2 = 1.
        self.N = 0
        self.packet_rate_Q2 = self.N/self.timer
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

        self.scheduled_transmissions = [0, 1]


        # Statistics
        self.reward_interval = 100
        self.count_packets_Q2_to_D2 = 0
        self.delay_Q1 = []

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
    def get_Pr_suc_rx_Q1_to_D1(self, power1, W_tx_Q2):
        power2 = self._calculate_Q2_tx_power(power1, W_tx_Q2)
        if power1 > self.threshold1*power2:
            # Equation (7) first part (before the multiplication symbol X)
            if W_tx_Q2 == True:  
                Pr_suc_rx_Q1_to_D1 = np.exp(-(self.threshold1 * self.distance1**self.PathLoss_to_D1)/(power1 - self.threshold1 * power2) * (1 + self.power_J*self.g**2))
            else: 
                Pr_suc_rx_Q1_to_D1 = np.exp(((-self.threshold1 * self.distance1**self.PathLoss_to_D1)/power1) *(1 + self.power_J*self.g**2))#no jamming.
        else:
            Pr_suc_rx_Q1_to_D1 = .0
        return Pr_suc_rx_Q1_to_D1
    
    # Probability that D2 will successfuly decode the packet sent by Q2. The computation depends on whether Q1 transmits or not.
    def get_Pr_suc_rx_Q2_to_D2(self, power1, W_tx_Q1,  W_tx_Q2):
        power2 = self._calculate_Q2_tx_power(power1, W_tx_Q2)
        if power2 > self.threshold2*power1:
            if W_tx_Q1 == True:       # Jammingself.PathLoss
                Pr_suc_rx_Q2_to_D2 =  np.exp(-(self.threshold2 * self.distance2**self.PathLoss_to_D2)/(power2 - self.threshold2*power1))/(1 + (self.threshold2 * self.power_J/(power2-self.threshold2*power1))*((self.distance2/self.distance3)**self.PathLoss_to_D2))
            else:                           # No jamming
                Pr_suc_rx_Q2_to_D2 =  np.exp(-(self.threshold2 * self.distance2**self.PathLoss_to_D2)/power2) 
        else:
            Pr_suc_rx_Q2_to_D2 = .0
        return Pr_suc_rx_Q2_to_D2

    # Probability that D2 will successfuly decode the packet sent by Q1. This is the secrecy violation scenario!
    def get_Pr_suc_rx_Q1_to_D2(self, power1, W_tx_Q2):
        power2 = self._calculate_Q2_tx_power(power1, W_tx_Q2)
        if power1 > self.threshold1*power2:
            if W_tx_Q2 == True:
                Pr_suc_rx_Q1_to_D2 =  np.exp(-(self.threshold1 * self.distance2**self.PathLoss_to_D2)/(power1 - self.threshold1*power2))/(1 + (self.threshold1*self.power_J)/(power1 - self.threshold1*power1)*(self.distance2/self.distance3)**self.PathLoss_to_D2)
            else:
                Pr_suc_rx_Q1_to_D2 =  np.exp(-self.threshold2 * self.distance2**self.PathLoss_to_D2/power1)/(1 + self.threshold2*self.power_J/power1*((self.distance2/self.distance3)**self.PathLoss_to_D2))
        else:
            Pr_suc_rx_Q1_to_D2 = .0
        return Pr_suc_rx_Q1_to_D2

    def compute_reward(self, power1, W_tx_Q1, W_tx_Q2, w_suc_rx_Q1_to_D1, w_suc_rx_Q2_to_D2, w_suc_rx_Q1_to_D2):
        reward = .0
        if W_tx_Q1 == True and W_tx_Q2 == True:
            if w_suc_rx_Q1_to_D1 and w_suc_rx_Q2_to_D2:
                reward += 2
                if (not w_suc_rx_Q1_to_D2):
                    reward += 10
            elif w_suc_rx_Q1_to_D1 and not w_suc_rx_Q2_to_D2:
                reward += 1
                if (not w_suc_rx_Q1_to_D2):
                    reward += 10
            elif not w_suc_rx_Q1_to_D1 and w_suc_rx_Q2_to_D2:
                reward += 1
            else:
                reward += .0
        elif W_tx_Q1 == True and W_tx_Q2 == False and w_suc_rx_Q1_to_D1: # and (not w_suc_rx_Q1_to_D2):
            reward += 1.
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
        w_1 = 0.5
        secrecy_reward = .0
        if w_suc_rx_Q1_to_D1 == True and w_suc_rx_Q1_to_D2 == False:
            secrecy_reward = 1.0
        
        return w_1*(1 - float(self.Q1.backlog)/self.Q1.capacity) + (1-w_1)*float(self.packet_rate_Q2) + secrecy_reward
        #return (1-w_1)*float(self.packet_rate_Q2)


    def step(self, power1):
        W_tx_Q1 = self.scheduled_transmissions[0]
        W_tx_Q2 = self.scheduled_transmissions[1]

        # Calculate probabilities to have a successful reception at the destinations and the 
        # potential violation of the security constraint.
        Pr_suc_rx_Q1_to_D1 = .0
        Pr_suc_rx_Q1_to_D2 = .0
        if W_tx_Q1 == True:
            Pr_suc_rx_Q1_to_D1 = self.get_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2)
            Pr_suc_rx_Q1_to_D2 = self.get_Pr_suc_rx_Q1_to_D2(power1, W_tx_Q2)
        
        Pr_suc_rx_Q2_to_D2 = .0
        if W_tx_Q2 == True:
            Pr_suc_rx_Q2_to_D2 = self.get_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1, W_tx_Q2)

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
        self.packet_rate_Q2 = self.N/self.rate_interval

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

        #self.Q1.get_backlog()
        # if w_suc_rx_Q2_to_D2 == True:
        #     self.count_packets_Q2_to_D2 += 1

        # if w_suc_rx_Q2_to_D2 == True:
        #     self.count_packets_Q2_to_D2 += 1
        
        # if self.timer % self.reward_interval == 0:
        #     rate_reward = self.count_packets_Q2_to_D2/self.reward_interval

        #backlog_reward = 1/(new_state+1)
        #reward = secrecy_reward #+ backlog_reward
        #float(self.scheduled_transmissions[0]), float(self.scheduled_transmissions[1]),
        new_state = [float(self.Q1.backlog/self.Q1.capacity), float(self.packet_rate_Q2)]

        # Reward
        #reward = self.compute_reward(power1, W_tx_Q1, W_tx_Q2, w_suc_rx_Q1_to_D1, w_suc_rx_Q2_to_D2, w_suc_rx_Q1_to_D2)
        reward = self.compute_transition_reward(new_state, w_suc_rx_Q1_to_D1, w_suc_rx_Q1_to_D2)

        if self.timer > self.rate_interval:
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
        self.packet_rate_Q2 = self.N/self.rate_interval
        return  [float(self.Q1.backlog/self.Q1.capacity), float(self.packet_rate_Q2)]
