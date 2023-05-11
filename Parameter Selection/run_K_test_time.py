import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from test_functions import *

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


# T = 100000 #Number of Total Iteration
J_sst = 10  # Number of Cohorts
L_sst = 25  # Number of Treatments
n_sst_list = [5000, 20000]

#K_list_sst = [50,75,100,150,200,int(n_sst)]  # 6 Instance
var_scale_sst = 0.2
print_opt = 1

repeats_sst = 20  # Currently we don't take average here
m_sst = 20
time_cap_sst = 1000
time_freq_sst = 20
dual_gap_freq_sst = 10

#Create Emp_dist





for n_num in n_sst_list:
    data_list = []
    K_list_sst = [100, int(n_num)]
    #Implement K_test_time
    for rep_idx in range(repeats_sst):
        temp_list = []
        mean_array_sst = np.random.rand(m_sst, J_sst, L_sst) / J_sst
        # Create our dataset
        emp_dist_value = np.zeros([m_sst, J_sst, L_sst, n_num])
        emp_dist_value[:] = np.nan
        x_0_sst = np.ones([J_sst, L_sst]) / L_sst
        p_0_sst = np.ones([m_sst, n_num]) / n_num
        RHS_sst = np.zeros(m_sst)
        for i in range(m_sst):
            RHS_sst[i] = np.sum(np.mean(mean_array_sst, axis=2)[i, :]) * 0.95

        # #Simple data generation
        # emp_dist_value_1 = np.random.normal(loc=0.5, scale=0.125, size=(m_sst, J_sst, int(L_sst / 2), n_num))
        # emp_dist_value_2 = np.random.normal(loc=0.25, scale=0.125, size=(m_sst, J_sst, int((L_sst + 1) / 2), n_num))
        # emp_dist_value = np.concatenate([emp_dist_value_1, emp_dist_value_2], axis=2)

        # Random mean data generation
        emp_dist_value = np.zeros([m_sst, J_sst, L_sst, n_num])
        emp_dist_value[:] = np.nan
        for m_idx in range(m_sst):
            for j_idx in range(J_sst):
                for l_idx in range(L_sst):
                    emp_dist_value[m_idx, j_idx, l_idx, :] = np.random.normal(
                        loc=mean_array_sst[m_idx, j_idx, l_idx],
                        scale=var_scale_sst * mean_array_sst[m_idx, j_idx, l_idx], size=n_num)

        for k_idx in range(len(K_list_sst)):
            stat = K_test_time(J_sst, L_sst,n_num, m_sst,emp_dist_value, RHS_sst, time_cap_sst, time_freq_sst,\
                                    dual_gap_freq_sst,K_list_sst[k_idx],print_opt)
            temp_list = [n_num, J_sst * L_sst, m_sst, K_list_sst[k_idx], time_cap_sst, time_freq_sst]
            temp_list.append(stat.dual_gap_list[0])
            data_list.append(temp_list)
    df = pd.DataFrame(data_list, columns=['n', 'd', 'm', 'K', 'time cap', 'time freq', 'Avg_Dual_Gap'])
    custom_path = '/results/K_time_result/K_test_time_n={}_d={}_m={}.csv'.format(n_num, J_sst * L_sst, m_sst)
    save_path = parent_path + custom_path
    df.to_csv(save_path, index=False)

# #Save this data into csv file
#
# dual_len = len(result_list[0][0].dual_gap_list[0])
# dual_gap_arr = np.zeros((repeats_sst,len(K_list_sst), dual_len))
#
# for rep_idx in range(repeats_sst):
#     for k_idx in range(len(K_list_sst)):
#         dual_gap_arr[rep_idx,k_idx,:] = result_list[rep_idx][k_idx].dual_gap_list[0]
#
# avg_dual_gap = np.average(dual_gap_arr,axis = 0)



#Save in csv

# data_list = []
# for rep_idx in range(repeats_sst):
#     for k_idx in range(len(K_list_sst)):
#
#         temp_list.append(avg_dual_gap[k_idx,:].tolist())
#         temp_list.append([])
#         data_list.append(temp_list)
#
# temp_list = [n_sst, J_sst * L_sst, m_sst, n_sst, time_cap_sst, time_freq_sst]
# temp_list.append(FMD_stat.dual_gap_list[0])
# temp_list.append(time_stamp_list)
# data_list.append(temp_list)





