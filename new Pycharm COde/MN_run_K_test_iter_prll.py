import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from test_functions import *
import multiprocessing as mp

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

if __name__ == '__main__':

    current_path = os.getcwd()
    parent_path = os.path.dirname(current_path)

    J_sst = 10  # Number of Cohorts
    L_sst = 25  # Number of Treatments


    n_sst = 20000
    K_list_sst = [100,400]
    print_opt = 1

    repeats_sst = 2  # Currently we don't take average here
    m_sst = 10
    time_cap_sst = 300
    time_freq_sst = 30
    dual_gap_freq_sst = 600

    #Create Emp_dist
    var_scale_sst = 0.1
    mean_array_sst = np.random.rand(m_sst, J_sst, L_sst) / J_sst
    # Create our dataset
    emp_dist_value = np.zeros([m_sst, J_sst, L_sst, n_sst])
    emp_dist_value[:] = np.nan

    RHS_sst = np.zeros(m_sst)
    for i in range(m_sst):
        RHS_sst[i] = np.sum(np.mean(mean_array_sst, axis=2)[i, :]) * 1.1

    # Random mean data generation
    emp_dist_value = np.zeros([m_sst, J_sst, L_sst, n_sst])
    emp_dist_value[:] = np.nan
    for m_idx in range(m_sst):
        for j_idx in range(J_sst):
            for l_idx in range(L_sst):
                emp_dist_value[m_idx, j_idx, l_idx, :] = np.random.normal(
                    loc=mean_array_sst[m_idx, j_idx, l_idx],
                    scale=var_scale_sst * mean_array_sst[m_idx, j_idx, l_idx], size=n_sst)



    result_list = []
    pool = mp.Pool(mp.cpu_count())
    result_objects = [pool.apply_async(K_test_time, args = (J_sst, L_sst, n_sst, m_sst, emp_dist_value, RHS_sst, time_cap_sst,\
                time_freq_sst,K_list_sst[k_idx], print_opt)) for rep_idx in range(repeats_sst) for k_idx in range(len(K_list_sst))]
    result = [r.get() for r in result_objects]
    pool.close()
    pool.join()

    FMD_stat, time_stamp_list = K_test_time_FMD(J_sst, L_sst, n_sst, m_sst, emp_dist_value, RHS_sst, \
                                                time_cap_sst, dual_gap_freq_sst, print_opt)


    result_sorted = [] #result_sorted[k_idx][rep_idx]
    for k_idx in range(len(K_list_sst)):
        result_sorted.append([])

    # Resort the output
    for res_idx in range(len(result)):
        for k_idx in range(len(K_list_sst)):
            if result[res_idx][1] == int(K_list_sst[k_idx]):
                result_sorted[k_idx].append(result[res_idx][0])

    data_list = []
    dual_len = len(result_sorted[0][0].dual_gap_list[0])
    dual_gap_arr = np.zeros((repeats_sst, len(K_list_sst), dual_len))

    for k_idx in range(len(K_list_sst)):
        temp_data_list = [n_sst, J_sst * L_sst, m_sst, K_list_sst[k_idx], time_cap_sst, time_freq_sst]
        for rep_idx in range(repeats_sst):
            stat = result_sorted[k_idx][rep_idx]
            dual_gap_arr[rep_idx,k_idx,:] = stat.dual_gap_list[0]
        avg_dual_gap = np.average(dual_gap_arr,axis = 0)
        temp_data_list.append(avg_dual_gap[k_idx,:].tolist())
        temp_data_list.append([])
        data_list.append(temp_data_list)

    #Add FMD result
    temp_data_list = [n_sst, J_sst * L_sst, m_sst, n_sst, time_cap_sst, time_freq_sst]
    temp_data_list.append(FMD_stat.dual_gap_list[0])
    temp_data_list.append(time_stamp_list)
    data_list.append(temp_data_list)

    df= pd.DataFrame(data_list, columns = ['n','d','m','K','time cap', 'time freq', 'Avg_Dual_Gap','time stamp'])
    save_path = parent_path + '/results/K_time_result/prll_K_test_time_n=' +str(n_sst) + 'd=' + str(J_sst * L_sst) + '_m=' + str(m_sst) + '.csv'
    df.to_csv(save_path, index=False)

