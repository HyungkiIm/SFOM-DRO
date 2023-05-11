import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
wrk_path = parent_path + '/results/K_time_result/'



n = 45000
poly_degree = 4
file_name = "FML_K_time_n={}.pdf".format(n)
fig_save_path = '../../../' + '\LaTeX Notes\Figures\FairnessML'


df = pd.read_csv(wrk_path + 'K_test_time_n={}_poly={}_rdix=1.csv'.format(n,poly_degree))
df['Avg_Dual_Gap'] = df['Avg_Dual_Gap'].apply(json.loads)
time_cap = df['time cap'][0]
time_freq = df['time freq'][0]
x_axis = [0]
iter_count = 1
for i in range(int(time_cap / time_freq)):
    x_axis.append(iter_count * time_freq)
    iter_count += 1
df_len = len(df)
dual_len = len(x_axis)
df_K = df.loc[df["K"] == 200]
df_n = df.loc[df['K'] == n]
K_index_list = df_K.index
n_index_list = df_n.index

K_log_dual_gap_arr = np.zeros([len(df_K),dual_len])
n_log_dual_gap_arr = np.zeros([len(df_n),dual_len])

temp_idx = 0
for idx in K_index_list:
    K_log_dual_gap_arr[temp_idx,:] = np.log(np.asarray(df_K['Avg_Dual_Gap'][idx]))
    temp_idx += 1
temp_idx = 0
for idx in n_index_list:
    n_log_dual_gap_arr[temp_idx,:] = np.log(np.asarray(df_n['Avg_Dual_Gap'][idx]))
    temp_idx += 1


K_mean_arr = np.mean(K_log_dual_gap_arr, axis = 0)
K_std_arr = np.std(K_log_dual_gap_arr, axis = 0)
n_mean_arr = np.mean(n_log_dual_gap_arr, axis = 0)
n_std_arr = np.std(n_log_dual_gap_arr, axis = 0)


x_tick_list = [10*i for i in range(20)]
fig = plt.figure(figsize = (16,8))
plt.rcParams.update({'font.size': 25})
plt.xticks(x_tick_list)
plt.ylabel('log(SP Gap)')
plt.xlabel('cpu time')
#plt.title('n = %s' % n)
iter = 25

plt.plot(x_axis[:iter], K_mean_arr[:iter], label='SOFO', color = 'blue')
plt.fill_between(x_axis[:iter],(K_mean_arr - 1.96*K_std_arr)[:iter],\
                (K_mean_arr + 1.96*K_std_arr)[:iter],color = 'blue', alpha= 0.1)
plt.plot(x_axis[:iter], n_mean_arr[:iter], label='OFO', color = 'red')
plt.fill_between(x_axis[:iter],(n_mean_arr - 1.96*n_std_arr)[:iter],\
                (n_mean_arr + 1.96*n_std_arr)[:iter],color = 'red', alpha= 0.1)

plt.legend(loc="upper right",prop={'size': 25})
plt.tick_params(axis = 'y',direction='in')
plt.tick_params(axis = 'x',direction='in')
os.chdir(fig_save_path)
plt.savefig(file_name,dpi=600)
plt.show()
