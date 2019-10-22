import pandas as pd
import matplotlib.pyplot as plt


dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\e-learning_categorical_with_teacher.csv'
results_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis'
df = pd.read_csv(dataset_path)
categ_cols = ['tutor_mode', 'answer_type', 'type', 'school_id']

del df['user_id']
# del df['teacher_id']

# for col in df.columns:
# # for col in categ_cols:
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
#             bins = 12
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
#     # else:
#     #     df[col].value_counts().plot(kind='bar')
#     #     # plt.savefig(results_path + '\\' + col + '.png')
#     #     plt.title(col)
#     #     plt.show()

id = 57418
teacher = df.loc[df['teacher_id'] == 57418]
print('teacher '+str(id)+', '+str(len(teacher))+' instances')
print('MEANS:\n'+str(teacher.mean(axis = 0, skipna = True)))
print('STDS:\n'+str(teacher.std(axis = 0, skipna = True)))


