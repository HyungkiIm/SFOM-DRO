import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from FML_test_functions import *
from sklearn.model_selection import train_test_split

#Todo
"""
1. We need to fix this code after changing K_test function on test_functions.py
2. Change x-axis of K_test plot to computation time.
"""

"""

Plot 3: Testing convergence on different K value

Here we run for fixed T_cap, However we use optimal step-size for each case with very low epsilon 1e-8, so as to not
terminate before T_cap. 

"""
current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
data_path = parent_path + '/data/adult/'

#Read Adult Data
poly_degree = 4
df = pd.read_csv(data_path + 'adult_processed_poly={}.csv'.format(poly_degree), index_col=0)
df_np = df.to_numpy()

#Only use the first 3e4 samples

X = df_np[:,:-1]
y = df_np[:,-1]
n_samples = X.shape[0]

seed_sst = 1234
np.random.seed(seed_sst)
#n_list_nt = np.linspace(10000,20000,11)
n_sst = 45000
c = 0.05 #Second and Third Constraint RHS

repeats_sst = 20
dual_gap_freq_sst = 5000
K_list_sst = [500,1000,5000]  # 6 Instance
#K_list_sst = [50,100]
T_cap_sst = 10000
print_opt = 1

# stat_list_sst[n_idx][rep_idx][alg_idx]

result_list = []
for k_sst in K_list_sst:
    for rep_idx in range(repeats_sst):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_sst / n_samples, shuffle=True, random_state=77)
        SMD_stat = K_test_iter(X_train, y_train, c, dual_gap_freq_sst, n_sst, K_sst, T_cap_sst, print_opt)
        result_list.append(stat_list)
#Save this data into csv form

#Get the length of each dual_gap_list
dual_len = len(result_list[0][0].dual_gap_list[0])
dual_gap_arr = np.zeros((repeats_sst,len(K_list_sst)+1, dual_len))
i_equal_count_arr = np.zeros((repeats_sst,len(K_list_sst)+1))
i_approx_count_arr = np.zeros((repeats_sst,len(K_list_sst)+1))
i_combined_count_arr = np.zeros((repeats_sst,len(K_list_sst)+1))

for rep_idx in range(repeats_sst):
    for k_idx in range(len(K_list_sst)):
        dual_gap_arr[rep_idx,k_idx,:] = result_list[rep_idx][k_idx].dual_gap_list[0]
        i_equal_count_arr[rep_idx, k_idx] = result_list[rep_idx][k_idx].i_equal_count
        i_approx_count_arr[rep_idx, k_idx] = result_list[rep_idx][k_idx].i_approx_count
        i_combined_count_arr[rep_idx, k_idx] = result_list[rep_idx][k_idx].i_combined

#avg_dual_gap = np.average(dual_gap_arr,axis = 0)
avg_i_equal_count = np.average(i_equal_count_arr, axis = 0)
avg_i_approx_count = np.average(i_approx_count_arr, axis = 0)
avg_i_combined_count = np.average(i_combined_count_arr, axis = 0)
avg_dual_gap = np.average(dual_gap_arr,axis=0)

#i_flag_count

#Save in csv

data_list = []
for k_idx in range(len(K_list_sst)):
    temp_list = [c, K_list_sst[k_idx], T_cap_sst, dual_gap_freq_sst]
    temp_list.append(avg_i_equal_count[k_idx])
    temp_list.append(avg_i_approx_count[k_idx])
    temp_list.append(avg_i_combined_count[k_idx])
    temp_list.append(avg_dual_gap[k_idx,:].tolist())
    data_list.append(temp_list)

df= pd.DataFrame(data_list, columns = ['c','K','Total_Iter', 'Dual_Freq','Avg_equal','Avg_approx', 'Avg_combined','Avg_Dual_gap'])
custom_path = '/results/K_iter_result/K_test_iter_n={}_c=_{}_poly={}.csv'.format(n_sst,c,poly_degree)
save_path = parent_path + custom_path
df.to_csv(save_path, index=False)

