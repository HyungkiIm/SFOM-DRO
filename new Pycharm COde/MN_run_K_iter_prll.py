import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import multiprocessing as mp
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

if __name__ == '__main__':


    current_path = os.getcwd()
    parent_path = os.path.dirname(current_path)


    # T = 100000 #Number of Total Iteration
    J_sst = 10  # Number of Cohorts
    L_sst = 25  # Number of Treatments


    dual_gap_freq_sst = 30
    n_sst = 5000
    #K_list_sst = [100,200]
    K_list_sst = [1,10,50,100,300, 500]  # 6 Instance
    T_cap_sst = 3000
    print_opt = 0

    repeats_sst = 20  # Currently we don't take average here
    m_sst = 40
    seed = 1234
    repeat_number = 0
    np.random.seed(1234 + repeat_number)





    result_list = []
    pool = mp.Pool(mp.cpu_count())
    result_objects = [pool.apply_async(K_test_iter, args = (J_sst, L_sst, dual_gap_freq_sst, n_sst, K_list_sst, T_cap_sst,
            m_sst, print_opt)) for rep_idx in range(repeats_sst)]
    result_list = [r.get() for r in result_objects]
    pool.close()
    pool.join()
    #Save this data into csv form

    #Get the length of each dual_gap_list
    dual_len = len(result_list[0][0].dual_gap_list[0])
    dual_gap_arr = np.zeros((repeats_sst,len(K_list_sst)+1, dual_len))
    i_flag_count_arr = np.zeros((repeats_sst,len(K_list_sst)+1))

    for rep_idx in range(repeats_sst):
        for k_idx in range(len(K_list_sst)+1):
            dual_gap_arr[rep_idx,k_idx,:] = result_list[rep_idx][k_idx].dual_gap_list[0]
            i_flag_count_arr[rep_idx,k_idx] = result_list[rep_idx][k_idx].i_flag_count

    avg_dual_gap = np.average(dual_gap_arr,axis = 0)
    avg_i_flag_count = np.average(i_flag_count_arr, axis = 0)

    #Save in csv

    data_list = []
    for k_idx in range(len(K_list_sst)):
        temp_list = [n_sst, J_sst * L_sst, m_sst,K_list_sst[k_idx], T_cap_sst, dual_gap_freq_sst]
        temp_list.append(avg_i_flag_count[k_idx])
        temp_list.append(avg_dual_gap[k_idx,:].tolist())
        data_list.append(temp_list)
    temp_list = [n_sst, J_sst * L_sst, m_sst,'FMD', T_cap_sst, dual_gap_freq_sst]
    temp_list.append(0)
    temp_list.append(avg_dual_gap[len(K_list_sst),:].tolist())
    data_list.append(temp_list)

    df= pd.DataFrame(data_list, columns = ['n','d','m','K','Total_Iter', 'Dual_Freq','Avg_i_flag', 'Avg_Dual_Gap'])
    save_path = parent_path + '/results/K_iter_result/K_iter_n=' +str(n_sst) + 'd=' + str(J_sst * L_sst) + '_m=' + str(m_sst) + '.csv'
    df.to_csv(save_path, index=False)

