import csv

import os
import pandas as pd
import matplotlib.pyplot as plt


# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\e-learning_full.csv'
# results_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis'
# df = pd.read_csv(dataset_path)
# categ_cols = ['tutor_mode', 'answer_type', 'type', 'school_id']
# user_directory = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis\\users'
# students = df.groupby(['user_id'])
# teachers = df.groupby(['teacher_id'])
# classes = df.groupby(['student_class_id'])

dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\KddCup\\2006\\kddCup_full_encoded.csv'
results_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\KddCup\\2006\\analysis'
df = pd.read_csv(dataset_path)
categ_cols = ['Problem Hierarchy']
user_directory = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\KddCup\\2006\\analysis\\users'
students = df.groupby(['Student'])

user_group = ''
# for users in [students, teachers, classes]:
for users in [students]:
    if user_group == '':
        user_group = 'students'
        continue
    elif user_group == 'students':
        user_group = 'teachers'
    else:
        user_group = 'classes'
    print(user_group)
    df = []
    user_count = 0
    for user_id, user_instances in users:
        user_count += 1
        df += [len(user_instances)]
    df = pd.DataFrame(df, columns=['len'])

    sub_df = df.loc[:]
    # if user_group == 'students':
    if user_group == 'students':
        # sub_df = df.loc[df['len'] <= 300]
        sub_df = df.loc[df['len'] <= 50]
    elif user_group == 'teachers':
        sub_df = df.loc[df['len'] <= 100]
    else:
        sub_df = df.loc[df['len'] <= 100]

    fig, ax = plt.subplots()
    mu = df.mean()
    sigma = df.std()
    textstr = '\n'.join((
        r'$\sigma=%.2f$' % (sigma,),
        r'$\mu=%.2f$' % (mu,),
        r'$\mathrm{n}=%.0f$' % (user_count,),
        r'$\mathrm{max}=%.0f$' % (df.max(),)))
    sub_df.hist(bins=100)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    plt.title(user_group + ' history length')
    plt.savefig(user_directory + '\\' + user_group + '.png')
    # plt.show()

# del df['user_id']
# del df['teacher_id']
# del df['student_class_id']
#
# # for col in df.columns:
# for col in categ_cols:
#     if col not in categ_cols:
#         print(col + ' [' +str(df[col].min()) + ', ' + str(df[col].max())+']')
#         sub_df = df
#
#         if col == 'ms_first_response':
#             sub_df = df.loc[(df['ms_first_response'] <= 150000) & (df['ms_first_response'] > 0)]
#             bins = 150
#
#         elif col == 'attempt_count':
#             sub_df = df.loc[(df['attempt_count'] <= 10)]
#             bins = 20
#
#         elif col == 'correct' or col == 'original' or col == 'first_action':
#             bins = 8
#
#         elif col == 'hint_count':
#             bins = 20
#
#         elif col == 'hint_total':
#             sub_df = df.loc[(df['hint_total'] <= 10)]
#             bins = 20
#
#         elif col == 'overlap_time':
#             sub_df = df.loc[(df['overlap_time'] <= 200000) & (df['overlap_time'] > 0)]
#             bins = 150
#
#         # else:
#         #     continue
#
#         # add stats
#         x = df[col]
#         fig, ax = plt.subplots()
#         mu = x.mean()
#         # median = np.median(x)
#         sigma = x.std()
#         textstr = '\n'.join((
#             r'$\mu=%.2f$' % (mu,),
#             # r'$\mathrm{med}=%.2f$' % (median,),
#             r'$\sigma=%.2f$' % (sigma,)))
#         sub_df.hist(column=col, bins=bins)
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#         plt.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
#                  verticalalignment='top', horizontalalignment='right', bbox=props)
#
#         plt.savefig(results_path + '\\' + col + '.png')
#
#     else:
#         df[col].value_counts().plot(kind='bar')
#         # plt.savefig(results_path + '\\' + col + '.png')
#         plt.title(col)
#         plt.show()

# id = 57418
# csv_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis\\teachers\\teacher_'+str(id)+'.csv'
# with open(csv_path, 'w', newline='') as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(['column', 'mean', 'std'])
#
#     rows = []
#     teacher = df.loc[df['teacher_id'] == 57418]
#     means = teacher.mean(axis = 0, skipna = True)
#     stds = teacher.std(axis = 0, skipna = True)
#     print('teacher '+str(id)+', '+str(len(teacher))+' instances')
#     print('MEANS:\n'+str(means))
#     print('STDS:\n'+str(stds))
#
#     for items in means.iteritems():
#         rows += [[items[0], items[1]]]
#
#     i = 0
#     for items in stds.iteritems():
#         writer.writerow(rows[i] + [items[1]])
#         i += 1
#
#     # dcsummary = pd.DataFrame([teacher.mean().round(5), teacher.std().round(5)], index=['Mean', 'STD'])
#     #
#     # # plt.table(cellText=dcsummary.values, colWidths=[0.25] * len(teacher.columns),
#     # plt.table(cellText=dcsummary.values,
#     #           rowLabels=dcsummary.index,
#     #           colLabels=dcsummary.columns,
#     #           cellLoc='center', rowLoc='center',
#     #           loc='top')
#     # plt.axis('off')
#     # plt.title('teacher '+str(id))
#     # plt.savefig(results_path + '\\teacher_' + str(id) + '.png')



