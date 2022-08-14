from struct import pack
import tensorflow as tf
from tensorflow.keras import layers

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import rl_phy_sec

def get_actor(num_states, lower_bound, upper_bound):
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

def get_packet_rates(rate_interval):
    _packet_rate_list = []
    for r_i in range(1, rate_interval + 1):
        for t in range(r_i):
            _packet_rate_list.append(t/r_i)
    packet_rate_list = (list(set(_packet_rate_list)))#.sort()
    packet_rate_list.sort()
    return packet_rate_list
    
def get_queue_utilization(capacity):
    queue_utilization_list = []
    for backlog in range(capacity+1):
        queue_utilization_list.append(backlog/capacity)
    return queue_utilization_list

def get_action(actor, queue_utilization, packet_rate, lower_bound, upper_bound):
    state = [queue_utilization, packet_rate]
    state = tf.expand_dims(tf.convert_to_tensor(state), 0)
    action = tf.squeeze(actor(state))
    legal_action = np.clip(action, lower_bound, upper_bound)
    return legal_action

def plot_actor_policy(actor, rate_interval, capacity, lower_bound, upper_bound):
    packet_rate_list = get_packet_rates(rate_interval)
    queue_utilization_list = get_queue_utilization(capacity)

    x_queue_utilization_list = []
    y_packet_rate_list = []
    z_action_list = []
    for q_u in queue_utilization_list:
        for p_r in packet_rate_list:
            x_queue_utilization_list.append(q_u)
            y_packet_rate_list.append(p_r)
            z_action_list.append(get_action(actor, q_u, p_r, lower_bound, upper_bound))
    
    x = np.asarray(x_queue_utilization_list).reshape(len(queue_utilization_list), len(packet_rate_list))
    y = np.asarray(y_packet_rate_list).reshape(len(queue_utilization_list), len(packet_rate_list))
    z = np.asarray(z_action_list).reshape(len(queue_utilization_list), len(packet_rate_list))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    angle = 60
    ax.view_init(30, angle)
    ax.plot_surface(y, y, z, cmap=cm.coolwarm)
    fig.savefig('policy.png')
    fig.show()

    return 0

def plot_heatmap_actor_policy(actor, rate_interval, capacity, lower_bound, upper_bound, figure_name):
    packet_rate_list = get_packet_rates(rate_interval)
    queue_utilization_list = get_queue_utilization(capacity)

    z_action_list = []
    for q_u in queue_utilization_list:
        for p_r in packet_rate_list:
            z_action_list.append(get_action(actor, q_u, p_r, lower_bound, upper_bound))
    
    x = np.asarray(queue_utilization_list)
    y = np.asarray(packet_rate_list)
    z = np.asarray(z_action_list).reshape(len(queue_utilization_list), len(packet_rate_list)).transpose()

    fig, ax = plt.subplots()
    plt.imshow(z,aspect='auto')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x)), labels=np.round(x, decimals=3).tolist())
    ax.set_yticks(np.arange(len(y)), labels=np.round(y, decimals=3).tolist())
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)

    # # Create colorbar
    plt.colorbar()

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(y)):
        for j in range(len(x)):
            text = ax.text(j, i, np.round(z[i, j], decimals=1), ha="center", va="center", color="w")
    
    ax.set_title("Actor Policy")
    plt.xlabel(f'Q_1 Utilization \n (backlog/capacity, where capacity={capacity})')
    plt.ylabel('Running packet rate \n (#Packets/time, where time=1, 2, ..., 10)')
    fig.tight_layout()
    fig.savefig(figure_name)
    fig.show()
    plt.close()

    return 0


if __name__ == '__main__':
    num_states = 2
    
    lambda_v, Pr_arrival_Q1, B_threshold, capacity_Q1, PathLoss_to_D1, PathLoss_to_D2, threshold1, threshold2, distance1,  distance2, distance3, power_max, power_J, g, q1, q2, P_max, packet_rate_interval, Q1_utilization_threshold, Q2_rate_threshold, TIN, SD = rl_phy_sec.define_parameters()
    
    lower_bound = (threshold1 / (1 + threshold1))*P_max
    upper_bound = (1/(1 + threshold2))*P_max

    rate_interval = 10
    actor = get_actor(num_states, lower_bound, upper_bound)
    actor.load_weights("tx_power_actor.h5")
    
    target_actor = get_actor(num_states, lower_bound, upper_bound)
    target_actor.load_weights("tx_power_target_actor.h5")
    
    # plot_actor_policy(actor, rate_interval, capacity_Q1, lower_bound, upper_bound)
    plot_heatmap_actor_policy(actor, rate_interval, capacity_Q1, lower_bound, upper_bound, f'actor_policy.png')
    # plot_heatmap_actor_policy(target_actor, rate_interval, capacity_Q1, lower_bound, upper_bound, 'target_actor_policy.png')

    # plot_actor_policy()
