import csv
import matplotlib as mpl
import matplotlib.pyplot as plt

episode_total_reward_list = []
episode_total_reward = .0
episode = 1
with open('training_logfile.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
            episode_steps = 0
        else:
            # print(f'\t{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]}')
            if row[0] == str(episode):
                episode_total_reward += float(row[4])
                episode_steps += 1
            else:# row[0] == episode + 1
                episode_total_reward_list.append(episode_total_reward)
                episode_total_reward = 0
                episode_steps = 0
                episode += 1
            line_count += 1
    print(f'Processed {line_count} lines.')
    print(episode_total_reward_list)

#y = [_y/episode_steps for _y in episode_total_reward_list]
y = [_y for _y in episode_total_reward_list]
x = [i for i in range(1,len(y)+1)]
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('Total reward')
# plt.xlim([-1, 2])
# plt.ylim([lower_bound, upper_bound])
# plt.show()
plt.savefig('total_reward_vs_episode.png')
plt.close()


