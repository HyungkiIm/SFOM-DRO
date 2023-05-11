import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
wrk_path = parent_path + '/results/n_num_result/'


n_list = np.linspace(10000,45000,15,dtype = int)
poly_degree = 3
K=200
fig_save_path = '../../../' + '\LaTeX Notes\Figures\FairnessML'
file_name = "FML_ntest_poly={}.pdf".format(poly_degree)

df_temp = pd.read_csv(wrk_path + 'n={}_K={}_poly={}.csv'.format(n_list[0],K,poly_degree))
repeat = len(df_temp)

data_list = []
for n_idx in range(n_list.shape[0]):
    temp_list = [n_list[n_idx]]
    df_temp = pd.read_csv(wrk_path + 'n={}_K={}_poly={}.csv'.format(n_list[n_idx],K,poly_degree))
    temp_list.append(np.mean(df_temp['SMD_Solve_time']))
    temp_list.append(np.std(df_temp['SMD_Solve_time']))
    temp_list.append(np.mean(df_temp['SMD_Iter']))
    temp_list.append(np.mean(df_temp['SMD_Max_Iter']))
    temp_list.append(np.mean(df_temp['FMD_Solve_time']))
    temp_list.append(np.std(df_temp['FMD_Solve_time']))
    temp_list.append(np.mean(df_temp['FMD_Iter']))
    temp_list.append(np.mean(df_temp['FMD_Max_Iter']))
    data_list.append(temp_list)

df = pd.DataFrame(data_list, columns = ['n','SMD_Avg_Solve_time','SMD_std_Solve_time', 'SMD_Avg_Iter', 'SMD_Max_Iter',\
                'FMD_Avg_Solve_time','FMD_std_Solve_time', 'FMD_Avg_Iter', 'FMD_Max_Iter'])
save_path = parent_path + '/results/n_num_result/Combined2_' + 'n={}_K={}_poly={}.csv'.format(n_list[0],K,poly_degree)
df.to_csv(save_path,index = False)

n_list = n_list/1000

fig = plt.figure(figsize = (16,8))
plt.rcParams.update({'font.size': 25})
plt.ylabel('time (s)')
plt.xlabel(r'n ($\times 10^3$)')
#plt.title('d = %s, m = %s' % (d,m))
plt.plot(n_list, df['SMD_Avg_Solve_time'].to_numpy(), label = "SOFO", color = 'blue',ls = '--')
plt.fill_between(n_list,df['SMD_Avg_Solve_time'].to_numpy() - 1.96*df['SMD_std_Solve_time'],\
                df['SMD_Avg_Solve_time'].to_numpy() + 1.96*df['SMD_std_Solve_time'],color = 'blue', alpha= 0.1)
plt.plot(n_list, df['FMD_Avg_Solve_time'].to_numpy(), label = "OFO", color = 'red')
plt.fill_between(n_list,df['FMD_Avg_Solve_time'].to_numpy() - 1.96*df['FMD_std_Solve_time'],\
                df['FMD_Avg_Solve_time'].to_numpy() + 1.96*df['FMD_std_Solve_time'],color = 'red', alpha= 0.1)
plt.scatter(n_list,df['SMD_Avg_Solve_time'].to_numpy(),marker = '^', color = 'blue')
plt.scatter(n_list,df['FMD_Avg_Solve_time'].to_numpy(),marker = '.', color = 'red')

x_tick = np.linspace(10,45,8).astype(int)
plt.legend(loc="upper left",prop={'size': 25})
plt.xticks(x_tick)
plt.tick_params(axis = 'y',direction='in')
plt.tick_params(axis = 'x',direction='in')
os.chdir(fig_save_path)
plt.savefig(file_name,dpi=600)

plt.show()