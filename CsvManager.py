import csv
import numpy as np
import os

def binarize(path, bin_count, diss_types):
    with open(path + ".csv", 'r', newline='') as file_in:
        with open(path + "_" + str(bin_count) + "_bins.csv", 'w', newline='') as file_out:
            reader = csv.reader(file_in)
            writer = csv.writer(file_out)

            # save rows, get min and max com
            rows = []
            in_header = True
            in_first = True
            min_com = 0
            max_com = 0
            base_acc = 0
            for row in reader:
                if in_header:
                    in_header = False
                    writer.writerow(row)
                    continue
                rows += [row]
                com = float(row[0])
                if in_first:
                    in_first = False
                    min_com = com
                    max_com = com
                    base_acc = float(row[1])
                    continue
                if com < min_com:
                    min_com = com
                if com > max_com:
                    max_com = com

            # init bins
            bin_width = (max_com - min_com) / bin_count
            bins = []
            for i in range(bin_count):
                bin = []
                for j in range(diss_types):
                    bin += [[]]
                bins += [bin]

            # fill bins
            for row in rows:
                com = float(row[0])
                i = int((com - min_com) / bin_width)
                if i == bin_count:
                    i = bin_count - 1
                for j in range(diss_types):
                    if row[2 + j] != "":
                        bins[i][j] += [(float(row[2 + j]))]

            # write file
            for i in range(len(bins)):
                bin = bins[i]
                row = [str(min_com + (i + 0.5) * bin_width), str(base_acc)]
                bin_empty = True
                for j in range(diss_types):
                    if len(bin[j]) != 0:
                        bin_empty = False
                        row += [str(np.mean(bin[j]))]
                    else:
                        row += [""]
                if not bin_empty:
                    writer.writerow(row)


def column_splitter(path, column, values_to_take=1):
    with open(path + ".csv", 'r', newline='') as file_in:
        with open(path + "_"+column+"_splitted.csv", 'w', newline='') as file_out:
            reader = csv.reader(file_in)
            writer = csv.writer(file_out)
            first = True
            column_idx = 0
            skill_name_column = 0
            user_id_column = 0
            skills_dict = {}
            for row in reader:
                if first:
                    first = False
                    writer.writerow(['user_id', 'original_'+column, column])
                    for i in range(len(row)):
                        name = row[i]
                        if name == column:
                            column_idx = i
                        elif name == 'user_id':
                            user_id_column = i
                        # elif name == 'skill_name':
                        #     skill_name_column = i
                else:
                    full_value = row[column_idx]
                    values = full_value.split(',')
                    # skill_names_string = row[skill_name_column]
                    # skill_names = skill_names_string.split(',')
                    user_id = row[user_id_column]
                    for i in range(len(values)):
                        if i == values_to_take:
                            break
                        value = values[i]
                        # skill_name = skill_names[i]
                        # if skill_id not in skills_dict:
                        #     skills_dict[skill_id] = skill_name
                        writer.writerow([user_id, full_value, value])

    # with open(path + "_skill_dictionary.csv", 'w', newline='') as file_out:
    #     writer = csv.writer(file_out)
    #     writer.writerow(['skill_id', 'skill_name'])
    #     for skill_id, skill_name in skills_dict.items():
    #         writer.writerow([skill_id, skill_name])


def manage_csv_line_delimited():
    path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\matific\\episode_run_0005_part_00'
    with open(path + ".csv", 'r', newline='') as file_in:
        with open(path + "_fixed.csv", 'w', newline='') as file_out:
            writer = csv.writer(file_out)
            cols = ['account_id',
                    'episode_id',
                    'session_key',
                    'discriminator',
                    'user_type',
                    'class_id',
                    'teacher_id',
                    'grade_code',
                    'locale_id',
                    'episode_slug',
                    'envelop_version',
                    'envelope',
                    'start_time',
                    'finish_time',
                    'last_submitted_index',
                    'time_spent_sec',
                    'activity_context',
                    'score',
                    'time_finished_sec',
                    'is_finished',
                    'problem_id',
                    'correct_answers_percentage',
                    'last_discriminator',
                    'last_in_session',
                    'before_replay',
                    'etl_time',
                    'client_ip']
            writer.writerow(cols)
            cols_not_used = ['account_id',
                             'episode_id',
                             'session_key',
                             'discriminator',
                             'user_type',
                             'class_id',
                             'teacher_id',
                             'grade_code',
                             'locale_id',
                             'episode_slug',
                             'envelop_version',
                             'envelope',
                             'start_time',
                             'finish_time',
                             'last_submitted_index',
                             'time_spent_sec',
                             'activity_context',
                             'score',
                             'time_finished_sec',
                             'is_finished',
                             'problem_id',
                             'correct_answers_percentage',
                             'last_discriminator',
                             'last_in_session',
                             'before_replay',
                             'etl_time',
                             'client_ip']

            i = 0
            for row in file_in:
                i += 1
                if i > 1000:
                    break
                raw_values = row.split('|')
                values = (x[1:-1] for x in raw_values)
                writer.writerow(values)


def csv_splitter(path, columns, max_rows_per_part=500000):
    with open(path + ".csv", 'r', newline='', encoding='utf-8') as file_in:
        reader = csv.reader(file_in)
        first = True
        header = []
        part = 0
        done = False
        while not done:
            part += 1
            with open(path + "_part_"+str(part)+".csv", 'w', newline='', encoding='utf-8') as file_out:
                part_end_reached = False
                writer = csv.writer(file_out)
                first_in_part = True
                rows_in_part = 0
                while not part_end_reached:
                    try:
                        row = next(reader)[columns]
                        rows_in_part += 1
                        if first:
                            first = False
                            header = row
                        elif first_in_part:
                            first_in_part = False
                            writer.writerow(header)
                            if not row[22] == '':
                                writer.writerow(row)
                        else:
                            if not row[22] == '':
                                writer.writerow(row)
                        if rows_in_part == max_rows_per_part:
                            part_end_reached = True
                    except StopIteration:
                        done = True
                        break


def from_file_names_to_csv(folder_path, file_path):
    with open(file_path, 'w', newline='') as file_out:
        writer = csv.writer(file_out)
        writer.writerow(['user_id', 'cos_sim'])
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in files:
                strings = name.split('_')
                user_id = strings[4].split('.')[0]
                cos_sim = strings[1]
                writer.writerow([user_id, cos_sim])




# path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning_2012\\ASSISTments_2012"
# path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\not used\\non_skill_builder_data_new'
# binarize(path, 30, 3)
# column_splitter(path, 'opportunity')
# csv_splitter(path, [0, 3, 8, 9, 10, 14, 15, 16, 18, 20, 22, 23, 24, 25], 1000)

folder_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis\\users\\distributions\\all skills\\train_percent_50\\skill'
file_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis\\users\\distributions\\all skills\\train_percent_50\\cos_sim_of_sets.csv'

from_file_names_to_csv(folder_path, file_path)
