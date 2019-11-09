import csv

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial
from numpy import dot
from numpy.linalg import norm

import time
import numpy as np


# # dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\e-learning_full.csv'
# # results_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis'
# # df = pd.read_csv(dataset_path)
# # categ_cols = ['tutor_mode', 'answer_type', 'type', 'school_id']
# # user_directory = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis\\users'
# # students = df.groupby(['user_id'])
# # teachers = df.groupby(['teacher_id'])
# # classes = df.groupby(['student_class_id'])
#
# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\KddCup\\2006\\kddCup_full_encoded.csv'
# results_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\KddCup\\2006\\analysis'
# df = pd.read_csv(dataset_path)
# categ_cols = ['Problem Hierarchy']
# user_directory = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\KddCup\\2006\\analysis\\users'
# students = df.groupby(['Student'])
#
# user_group = ''
# # for users in [students, teachers, classes]:
# for users in [students]:
#     if user_group == '':
#         user_group = 'students'
#         continue
#     elif user_group == 'students':
#         user_group = 'teachers'
#     else:
#         user_group = 'classes'
#     print(user_group)
#     df = []
#     user_count = 0
#     for user_id, user_instances in users:
#         user_count += 1
#         df += [len(user_instances)]
#     df = pd.DataFrame(df, columns=['len'])
#
#     sub_df = df.loc[:]
#     # if user_group == 'students':
#     if user_group == 'students':
#         # sub_df = df.loc[df['len'] <= 300]
#         sub_df = df.loc[df['len'] <= 50]
#     elif user_group == 'teachers':
#         sub_df = df.loc[df['len'] <= 100]
#     else:
#         sub_df = df.loc[df['len'] <= 100]
#
#     fig, ax = plt.subplots()
#     mu = df.mean()
#     sigma = df.std()
#     textstr = '\n'.join((
#         r'$\sigma=%.2f$' % (sigma,),
#         r'$\mu=%.2f$' % (mu,),
#         r'$\mathrm{n}=%.0f$' % (user_count,),
#         r'$\mathrm{max}=%.0f$' % (df.max(),)))
#     sub_df.hist(bins=100)
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     plt.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
#              verticalalignment='top', horizontalalignment='right', bbox=props)
#     plt.title(user_group + ' history length')
#     plt.savefig(user_directory + '\\' + user_group + '.png')
#     # plt.show()
#
# # del df['user_id']
# # del df['teacher_id']
# # del df['student_class_id']
# #
# # # for col in df.columns:
# # for col in categ_cols:
# #     if col not in categ_cols:
# #         print(col + ' [' +str(df[col].min()) + ', ' + str(df[col].max())+']')
# #         sub_df = df
# #
# #         if col == 'ms_first_response':
# #             sub_df = df.loc[(df['ms_first_response'] <= 150000) & (df['ms_first_response'] > 0)]
# #             bins = 150
# #
# #         elif col == 'attempt_count':
# #             sub_df = df.loc[(df['attempt_count'] <= 10)]
# #             bins = 20
# #
# #         elif col == 'correct' or col == 'original' or col == 'first_action':
# #             bins = 8
# #
# #         elif col == 'hint_count':
# #             bins = 20
# #
# #         elif col == 'hint_total':
# #             sub_df = df.loc[(df['hint_total'] <= 10)]
# #             bins = 20
# #
# #         elif col == 'overlap_time':
# #             sub_df = df.loc[(df['overlap_time'] <= 200000) & (df['overlap_time'] > 0)]
# #             bins = 150
# #
# #         # else:
# #         #     continue
# #
# #         # add stats
# #         x = df[col]
# #         fig, ax = plt.subplots()
# #         mu = x.mean()
# #         # median = np.median(x)
# #         sigma = x.std()
# #         textstr = '\n'.join((
# #             r'$\mu=%.2f$' % (mu,),
# #             # r'$\mathrm{med}=%.2f$' % (median,),
# #             r'$\sigma=%.2f$' % (sigma,)))
# #         sub_df.hist(column=col, bins=bins)
# #         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# #         plt.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
# #                  verticalalignment='top', horizontalalignment='right', bbox=props)
# #
# #         plt.savefig(results_path + '\\' + col + '.png')
# #
# #     else:
# #         df[col].value_counts().plot(kind='bar')
# #         # plt.savefig(results_path + '\\' + col + '.png')
# #         plt.title(col)
# #         plt.show()
#
# # id = 57418
# # csv_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis\\teachers\\teacher_'+str(id)+'.csv'
# # with open(csv_path, 'w', newline='') as csv_file:
# #     writer = csv.writer(csv_file)
# #     writer.writerow(['column', 'mean', 'std'])
# #
# #     rows = []
# #     teacher = df.loc[df['teacher_id'] == 57418]
# #     means = teacher.mean(axis = 0, skipna = True)
# #     stds = teacher.std(axis = 0, skipna = True)
# #     print('teacher '+str(id)+', '+str(len(teacher))+' instances')
# #     print('MEANS:\n'+str(means))
# #     print('STDS:\n'+str(stds))
# #
# #     for items in means.iteritems():
# #         rows += [[items[0], items[1]]]
# #
# #     i = 0
# #     for items in stds.iteritems():
# #         writer.writerow(rows[i] + [items[1]])
# #         i += 1
# #
# #     # dcsummary = pd.DataFrame([teacher.mean().round(5), teacher.std().round(5)], index=['Mean', 'STD'])
# #     #
# #     # # plt.table(cellText=dcsummary.values, colWidths=[0.25] * len(teacher.columns),
# #     # plt.table(cellText=dcsummary.values,
# #     #           rowLabels=dcsummary.index,
# #     #           colLabels=dcsummary.columns,
# #     #           cellLoc='center', rowLoc='center',
# #     #           loc='top')
# #     # plt.axis('off')
# #     # plt.title('teacher '+str(id))
# #     # plt.savefig(results_path + '\\teacher_' + str(id) + '.png')

def compare_history_train_and_history_test_sets():
    full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\user_skills.csv'
    save_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis\\users\\distributions\\all skills"
    target_col = 'correct'
    # categ_cols = ['tutor_mode', 'answer_type', 'type', 'skill']
    categ_cols = ['skill']
    user_group_names = ['user_id']
    train_frac = 0.5
    repetitions = [0]
    min_history_size = 20
    max_history_size = 300

    # df_full = pd.read_csv(full_dataset_path)[:10000]
    df_full = pd.read_csv(full_dataset_path)

    try:
        del df_full['school_id']
        del df_full['teacher_id']
        del df_full['student_class_id']
    except:
        pass

    print('pre-processing data and splitting into train and test sets...')

    # create user groups
    user_groups_train = []
    user_groups_test = []
    for user_group_name in user_group_names:
        user_groups_test += [df_full.groupby([user_group_name])]

    # separate histories into training and test sets
    students_group = user_groups_test[0]
    df_train = students_group.apply(lambda x: x[:int(len(x) * train_frac) + 1])
    df_train.index = df_train.index.droplevel(0)
    user_groups_test[0] = df_full.drop(df_train.index).groupby([user_group_names[0]])
    user_groups_train += [df_train.groupby([user_group_names[0]])]

    user_group_idx = -1

    for user_group_test in user_groups_test:
        user_group_idx += 1
        user_group_name = user_group_names[user_group_idx]

        print(user_group_name)

        total_users = 0
        # i = 0
        for user_id, test in user_group_test:
            # i += 1
            # print(str(i) + ' ' + str(user_id) + ' size='+str(len(test)))
            if len(test) >= min_history_size:
                total_users += 1

        print(str(total_users) + ' users')

        user_count = 0
        for user_id, test in user_group_test:
            if len(test) < min_history_size:
                continue
            user_count += 1
            print(str(user_count) + '/' + str(total_users) + ' ' + user_group_name + ' ' + str(
                user_id) + ', test = ' + str(len(test)))

            train = user_groups_train[user_group_idx - 1].get_group(user_id)

            # for col in df.columns:
            for col in categ_cols:
                if col not in categ_cols:
                    pass
                    # print(col + ' [' +str(df[col].min()) + ', ' + str(df[col].max())+']')
                    # sub_df = df
                    #
                    # if col == 'ms_first_response':
                    #     sub_df = df.loc[(df['ms_first_response'] <= 150000) & (df['ms_first_response'] > 0)]
                    #     bins = 150
                    #
                    # elif col == 'attempt_count':
                    #     sub_df = df.loc[(df['attempt_count'] <= 10)]
                    #     bins = 20
                    #
                    # elif col == 'correct' or col == 'original' or col == 'first_action':
                    #     bins = 8
                    #
                    # elif col == 'hint_count':
                    #     bins = 20
                    #
                    # elif col == 'hint_total':
                    #     sub_df = df.loc[(df['hint_total'] <= 10)]
                    #     bins = 20
                    #
                    # elif col == 'overlap_time':
                    #     sub_df = df.loc[(df['overlap_time'] <= 200000) & (df['overlap_time'] > 0)]
                    #     bins = 150
                    #
                    # # else:
                    # #     continue
                    #
                    # # add stats
                    # x = df[col]
                    # fig, ax = plt.subplots()
                    # mu = x.mean()
                    # # median = np.median(x)
                    # sigma = x.std()
                    # textstr = '\n'.join((
                    #     r'$\mu=%.2f$' % (mu,),
                    #     # r'$\mathrm{med}=%.2f$' % (median,),
                    #     r'$\sigma=%.2f$' % (sigma,)))
                    # sub_df.hist(column=col, bins=bins)
                    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    # plt.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    #          verticalalignment='top', horizontalalignment='right', bbox=props)
                    #
                    # plt.savefig(results_path + '\\' + col + '.png')
                else:
                    fig, ax = plt.subplots()
                    train_values = train[col].value_counts() / len(train)
                    test_values = test[col].value_counts() / len(test)
                    merged_df = pd.concat([train_values, test_values], axis=1).fillna(0)
                    merged_df.columns = ['train', 'test']

                    merged_df.plot.bar()
                    plt.legend(loc='upper right')
                    plt.xlabel(col + ' id')
                    plt.ylabel('fraction from set')
                    hist_len = len(train) + len(test)
                    title = user_group_name + '=' + str(user_id) + " col='" + col + "' train=" + str(int(100 * train_frac)) + '%'
                    plt.title(title)

                    train_vector = merged_df['train']
                    test_vector = merged_df['test']
                    cos_sim = dot(train_vector, test_vector) / (norm(train_vector) * norm(test_vector))
                    cos_sim_str = '%.5f' % (cos_sim,)
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    plt.text(0.05, 0.95, 'cos sim = '+cos_sim_str, transform=ax.transAxes,
                             verticalalignment='top', horizontalalignment='left', bbox=props)

                    directory = save_path + '\\train_percent_' + str(int(100 * train_frac)) + '\\' + col
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    plt.savefig(directory + '\\sim_' +cos_sim_str+'_'+user_group_name + '_' + str(user_id) + '.png')
                    # plt.show()
                    n = 100
                    if user_count % n == n - 1:
                        plt.close('all')


compare_history_train_and_history_test_sets()
