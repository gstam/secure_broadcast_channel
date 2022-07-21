import csv

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
        else:
            # print(f'\t{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]}')

            if row[0] == str(episode):
                episode_total_reward += float(row[4])
            else:# row[0] == episode + 1
                episode_total_reward_list.append(episode_total_reward)
                episode_total_reward = 0
                episode += 1
            line_count += 1
    print(f'Processed {line_count} lines.')
    print(episode_total_reward_list)


