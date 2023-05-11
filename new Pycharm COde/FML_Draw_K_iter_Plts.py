import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
wrk_path = parent_path + '/results/K_iter_result/'
n = 45000
k_list = [1,10,50,100,200,500]
poly_degree = 3
df = pd.read_csv(wrk_path + 'K={}_n={}_c=_0.05_poly={}.csv'.format(k_list[0],n,poly_degree))
repeat = len(df)
df['Avg_Dual_gap'] = df['Avg_Dual_gap'].apply(json.loads)
dual_len = len(df['Avg_Dual_gap'][0])
T = df['Total_Iter'][0]
dual_gap_freq = df['Dual_Freq'][0]
file_name = "FML_K_iter_poly={}.pdf".format(poly_degree)
fig_save_path = '../../../' + '\LaTeX Notes\Figures\FairnessML'

data_list = []
avg_equal_list = []
avg_approx_list = []
avg_combined_list = []
mean_list = []
std_list = []
dual_gap_arr = np.empty((repeat, dual_len))
for K in k_list:
    df = pd.read_csv(wrk_path + 'K={}_n={}_c=_0.05_poly={}.csv'.format(K, n, poly_degree))
    df['Avg_Dual_gap'] = df['Avg_Dual_gap'].apply(json.loads)
    for rep in range(repeat):
        dual_gap_arr[rep,:] = np.log(df['Avg_Dual_gap'][rep])
    mean_arr = np.mean(dual_gap_arr,axis=0)
    std_arr = np.std(dual_gap_arr,axis=0)
    # Stack starting duality gap (0.24606) to mean_arr and 0 std_arr
    mean_arr = np.insert(mean_arr,0,0.24606,axis=0)
    std_arr = np.insert(std_arr, 0, 0, axis=0)
    mean_list.append(mean_arr)
    std_list.append(std_arr)
    avg_equal_list.append(np.mean(df['Avg_equal']))
    avg_approx_list.append(np.mean(df['Avg_approx']))
    avg_combined_list.append(np.mean(df['Avg_combined']))
new_list = [k_list,avg_equal_list,avg_approx_list,avg_combined_list]
df_new = pd.DataFrame(new_list).transpose()
df_new.columns = ['K','Avg_equal','Avg_approx','Avg_combined']
save_path = parent_path + '/results/K_iter_result/Combined_n={}_poly={}.csv'.format(n,poly_degree)
df_new.to_csv(save_path,index=False)



x_axis =[0]
for t in range(T):
    if (t+2)%dual_gap_freq ==0:
        x_axis.append(t/1000)



fig = plt.figure(figsize = (16,8))
plt.rcParams.update({'font.size': 25})
plt.ylabel('log(SP Gap)')
plt.xlabel(r'iter ($\times 10^3$)')
#plt.title('n = %s Duality Gap' % n)
iter = 120
x_ticks = np.arange(11).astype(int)
for idx,K in enumerate(k_list):
    plt.plot(x_axis[:iter], mean_list[idx][:iter], label = 'K={}'.format(K))
    plt.fill_between(x_axis[:iter], mean_list[idx][:iter] - 1.96 * std_list[idx][:iter],
                     mean_list[idx][:iter] + 1.96 * std_list[idx][:iter],alpha=0.1)
plt.legend(loc="upper right",prop={'size': 20})
plt.xticks(x_ticks)
plt.tick_params(axis = 'y',direction='in')
plt.tick_params(axis = 'x',direction='in')
os.chdir(fig_save_path)
plt.savefig(file_name,dpi=600)
plt.show()
