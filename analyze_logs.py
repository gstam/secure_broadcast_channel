import os
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# An experiment is consisted of a series of episodes. 
# For each episode we get a total reward, i.e., a single value.
# For an experiment we get a list of total rewards.
# def get_list_of_total_rewards_for_experiment(data_folder):
#     episode_total_reward_list = []
#     episode_total_reward = .0
#     episode = 0
#     with open(os.path.join(data_folder, 'training_logfile.csv')) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=';')
#         line_count = 0
#         for row in csv_reader:
#             if line_count == 0:
#                 # print(f'Column names are {", ".join(row)}')
#                 line_count += 1
#                 episode_steps = 0
#             else:
#                 #print(f'\t{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]}')
#                 if row[0] == str(episode):
#                     episode_total_reward += float(row[4])
#                     episode_steps += 1
#                 else:# row[0] == episode + 1
#                     episode_total_reward_list.append(episode_total_reward)
#                     episode_total_reward = 0.0
#                     episode_steps = 0
#                     episode += 1
#                 line_count += 1
#         # print(f'Processed {line_count} lines.')
#         # print(episode_total_reward_list)

#     #y = [_y/episode_steps for _y in episode_total_reward_list]
#     y = [_y for _y in episode_total_reward_list]
#     x = [i for i in range(len(y))]
#     return x, y

def get_list_of_total_rewards_for_experiment(data_folder, episode_number):
    episode_total_reward_nparray = np.zeros(episode_number, dtype = float)
    with open(os.path.join(data_folder, 'training_logfile.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
                episode_steps = 0
            else:
                #print(f'\t{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]}')
                episode = int(row[0])
                episode_total_reward_nparray[episode] += float(row[4])
                line_count += 1

    return episode_total_reward_nparray


# A scenario is comprised of multiple experiments.
# Each experiment provides a list of total rewards where each single value corresponds to an episode.
# For a scenario we get a list of lists on the constraint that they are of equal length.
def get_list_of_total_episode_rewards_lists_for_scenario(scenario_folder, episode_number):
    scenario_rewards_list = []
    for experiment in os.scandir(scenario_folder):
        if experiment.is_dir():
            total_episode_reward_nparray = get_list_of_total_rewards_for_experiment(experiment.path, episode_number)
            if len(total_episode_reward_nparray) == episode_number:
                scenario_rewards_list.append(total_episode_reward_nparray)
    return scenario_rewards_list

def plot_scenario_results(scenario_folder, episode_number):
    scenario_rewards_list = get_list_of_total_episode_rewards_lists_for_scenario(scenario_folder, episode_number)
    scenario_rewards_matrix = np.array(scenario_rewards_list)
    y = np.mean(scenario_rewards_matrix, axis=0)
    x = [i for i in range(len(y))]
    y_std = np.std(scenario_rewards_matrix, axis=0)
    y_low = y - y_std/2
    y_high = y + y_std/2
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.fill_between(x, y_low, y_high, alpha=.5, linewidth=0)
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward')
    # plt.xlim([-1, 2])
    # plt.ylim([lower_bound, upper_bound])
    # plt.show()
    plt.savefig(os.path.join(scenario_folder, 'total_reward_vs_episode.png'))
    plt.close()
    return 0


def main(data_folder, episode_number):
    y = get_list_of_total_rewards_for_experiment(data_folder, episode_number)
    x = [i for i in range(len(y))]
    plt.plot(x, y)
    plt.xlabel('episode')
    plt.ylabel('Total reward')
    # plt.xlim([-1, 2])
    # plt.ylim([lower_bound, upper_bound])
    # plt.show()
    plt.savefig(os.path.join(data_folder, 'total_reward_vs_episode.png'))
    plt.close()

if __name__ == '__main__':
    scenario_folder = './Results/TIN/'
    episode_number = 60
    # get_rewards_for_scenario('./Results/TIN/')
    plot_scenario_results(scenario_folder, episode_number)
    # main('11_08_2022_08_11_36', episode_number)
