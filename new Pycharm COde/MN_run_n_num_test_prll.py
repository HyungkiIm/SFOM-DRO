import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import math
import time
import gurobipy as gp
from gurobipy import GRB
from statistics import mean
from tqdm import tqdm
from test_functions import n_num_test

#Todo
"""

"""

"""

Here we plot the computation time of SMD and FMD for different n values. 

Update Notes: 
Fixed plotting codes.


"""

if __name__ == '__main__':



    current_path = os.getcwd()
    parent_path = os.path.dirname(current_path)

    J_nt = 3  # Number of Cohorts
    L_nt = 5 # Number of Treatments

    n_list_nt = [1000,1200]
    #n_list_nt = np.linspace(10000,20000,11)
    m_nt = 5  # constraint number for figure 1
    repeats_nt = 2
    iter_max = 2
    K_nt = 100
    iter_count = 1

    for iter_idx in range(iter_max):
        data_list = []
        #async parallel
        pool = mp.Pool(mp.cpu_count())
        result_objects = [pool.apply_async(n_num_test, args = (int(n_list_nt[n_idx]),J_nt, L_nt, m_nt,K_nt,\
                                    1, 1)) for n_idx in range(len(n_list_nt)) for rep_idx in range(repeats_nt)]
        result = [r.get() for r in result_objects]
        pool.close()
        pool.join()

        result_sorted = []
        for n_idx in range(len(n_list_nt)):
            result_sorted.append([])

        #Resort the output
        for res_idx in range(len(result)):
            for n_idx in range(len(n_list_nt)):
                if result[res_idx][2] == int(n_list_nt[n_idx]):
                    result_sorted[n_idx].append(result[res_idx])

        #J_nt x L_nt
        for n_idx in range(len(n_list_nt)):
            temp_data_list = [n_list_nt[n_idx], J_nt * L_nt, m_nt]
            SMD_run_time_arr = np.zeros(repeats_nt)
            FMD_run_time_arr = np.zeros(repeats_nt)
            SMD_real_T_arr = np.zeros(repeats_nt)
            FMD_real_T_arr = np.zeros(repeats_nt)
            SMD_T_arr = np.zeros(repeats_nt)
            FMD_T_arr = np.zeros(repeats_nt)
            for rep_idx in range(repeats_nt):
                stat_SMD, stat_FMD, _ = result_sorted[n_idx][rep_idx]
                SMD_run_time_arr[rep_idx] = stat_SMD.total_solved_time
                FMD_run_time_arr[rep_idx] = stat_FMD.total_solved_time
                SMD_real_T_arr[rep_idx] = stat_SMD.real_T_list[0]
                FMD_real_T_arr[rep_idx] = stat_FMD.real_T_list[0]
                SMD_T_arr[rep_idx] = stat_SMD.max_iter
                FMD_T_arr[rep_idx] = stat_FMD.max_iter
            temp_data_list.append(np.mean(SMD_run_time_arr))
            temp_data_list.append(np.mean(SMD_real_T_arr))
            temp_data_list.append(np.mean(SMD_T_arr))
            temp_data_list.append(np.mean(FMD_run_time_arr))
            temp_data_list.append(np.mean(FMD_real_T_arr))
            temp_data_list.append(np.mean(FMD_T_arr))
            data_list.append(temp_data_list)
        df = pd.DataFrame(data_list, columns=['n', 'd', 'm', 'SMD_Avg_Solve_time', 'SMD_Avg_Iter', 'SMD_Max_Iter', \
                                              'FMD_Avg_Solve_time', 'FMD_Avg_Iter', 'FMD_Max_Iter'])
        save_path = parent_path + '/results/n_num_result/prr_n_test_d=' + str(J_nt * L_nt) + '_m=' + str(m_nt) +\
                    '_K=' + str(K_nt) + '_iter=' + str(iter_count) + '.csv'
        df.to_csv(save_path, index=False)
        iter_count += 1

#Merge csv files into one file

result_path = parent_path + '/results/n_num_result/' + str(J_nt * L_nt) + '_m=' + str(m_nt) +\
                    '_K=' + str(K_nt) + '_iter='

df_master = pd.read_csv(result_path + '0.csv')

print(df_master)

for idx in range(iter_max):
