import numpy as np
from fifo_queue import Queue

class Environment():

    def __init__(self, capacity, Pr_arrival_Q1, lambda_v, PathLoss, threshold1, threshold2, 
                    distance1, distance2, distance3, power_max, power_J, g, q1, q2, P_max):
        # Queue with Bernoulli arrivals and finite capacity.
        self.Q1 = Queue(capacity)
        self.Pr_arrival_Q1 = Pr_arrival_Q1
        
        # Saturated queue: It has a capacity of 1  but it never gets empty because we have a probability of 1 to get fresh arrival at each timeslot
        self.Q2 = Queue(capacity=1)
        self.Pr_arrival_Q2 = 1.

        self.Pr_tx_Q1 = q1
        self.Pr_tx_Q2 = q2
        self.lambda_v = lambda_v
        self.PathLoss = PathLoss
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
        if W_tx_Q1 == True:       # Jammingself.PathLoss
            Pr_suc_rx_Q2_to_D2 =  np.exp(-(self.threshold2 * self.distance2**3)/(power1 - self.threshold2*power1))*(1 + (self.threshold2 * self.power_J/(power2-self.threshold2*power1))*(self.distance2/self.distance3)**3)**(-1)
        else:                           # No jamming
            Pr_suc_rx_Q2_to_D2 =  np.exp(-(self.threshold2 * self.distance2**3)/power2) 
        return Pr_suc_rx_Q2_to_D2

    # Probability that D2 will successfuly decode the packet sent by Q1. This is the secrecy violation scenario!
    def get_Pr_suc_rx_Q1_to_D2(self, power1, W_tx_Q2):
        power2 = self.P_max - power1 
        if W_tx_Q2 == True:
            Pr_suc_rx_Q1_to_D2 =  np.exp(-(self.threshold1 * self.distance2**3)/(power1 - self.threshold1*power2))*(1 + self.threshold1*(self.power_J/(power1 - self.threshold1*power1))*(self.distance2/self.distance3)**3)**(-1)
        else:
            Pr_suc_rx_Q1_to_D2 =  np.exp(-(self.threshold1 * self.distance2**3)/(power1))*(1 + self.threshold1*(self.power_J/(power1))*(self.distance2/self.distance3)**3)**(-1)
        return Pr_suc_rx_Q1_to_D2

    def step(self, power1):
        # Get random transmissions from Q1 and Q2
        rnd = np.random.default_rng().uniform(0., 1., 1)
        W_tx_Q1 = False
        if self.Q1.backlog > 0 and rnd < self.Pr_tx_Q1:
                W_tx_Q1 = True
            
        rnd = np.random.default_rng().uniform(0., 1., 1)
        W_tx_Q2 = False
        if rnd < self.Pr_tx_Q2:
            W_tx_Q2 = True
        
        # Calculate probabilities to have a  successful reception at the destinations and the 
        # potential violation of the security constraint.
        Pr_suc_rx_Q1_to_D1 = .0
        Pr_suc_rx_Q1_to_D2 = .0
        if W_tx_Q1 == True:
            Pr_suc_rx_Q1_to_D1 = self.get_Pr_suc_rx_Q1_to_D1(power1, W_tx_Q2)
            Pr_suc_rx_Q1_to_D2 = self.get_Pr_suc_rx_Q1_to_D2(power1, W_tx_Q2)
        
        Pr_suc_rx_Q2_to_D2 = .0
        if W_tx_Q2 == True:
            Pr_suc_rx_Q2_to_D2 = self.get_Pr_suc_rx_Q2_to_D2(power1, W_tx_Q1)

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
        
        # Update state after successfull transmission.
        if w_suc_rx_Q1_to_D1:
            self.Q1.packet_departure()
        
        # Late arrivals at Q1
        rnd = np.random.default_rng().uniform(0., 1., 1)
        if rnd < self.Pr_arrival_Q1:
            self.Q1.packet_arrival()

        secrecy_reward = .0
        # Calculate Reward : Reward is provided only if 
        if W_tx_Q1 == True and W_tx_Q2 == True and w_suc_rx_Q1_to_D1 and (not w_suc_rx_Q1_to_D2): #and w_suc_rx_Q2_to_D2 
            secrecy_reward = 1.
        elif W_tx_Q1 == True and W_tx_Q2 == False and w_suc_rx_Q1_to_D1 and (not w_suc_rx_Q1_to_D2):
            secrecy_reward = 1.
        elif W_tx_Q1 == False and W_tx_Q2 == True and w_suc_rx_Q2_to_D2:
            secrecy_reward = 1.
        else:
            secrecy_reward = .0

        new_state = self.Q1.get_backlog()
        backlog_reward = 1/(new_state+1)
        
        reward = secrecy_reward #+ backlog_reward
        return reward, new_state

    def reset(self):
        self.Q1.set_backlog(0)
        return self.Q1.get_backlog()
