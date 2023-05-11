import os
import numpy as np
import pandas as pd
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


current_path = os.getcwd()
parent_path = os.path.dirname(current_path)

J_nt = 10  # Number of Cohorts
L_nt = 15 # Number of Treatments

n_list_nt = [5000,6000,7000,8000]

seed_nt = 1234
np.random.seed(seed_nt)
#n_list_nt = np.linspace(10000,20000,11)
m_nt = 5  # constraint number for figure 1
repeats_nt = 2
K_nt = 100
print_opt = 1

#data_list = []

#J_nt x L_nt
#temp_data_list = [n_list_nt[0], J_nt * L_nt, m_nt]
SMD_run_time_arr = np.zeros(repeats_nt)
FMD_run_time_arr = np.zeros(repeats_nt)
SMD_real_T_arr = np.zeros(repeats_nt)
FMD_real_T_arr = np.zeros(repeats_nt)
SMD_T_arr = np.zeros(repeats_nt)
FMD_T_arr = np.zeros(repeats_nt)
for n_num in n_list_nt:
    for rep_idx in range(repeats_nt):
        stat_SMD, stat_FMD, _ = n_num_test(int(n_num), J_nt, L_nt, m_nt, K_nt, print_opt = 1, feas_opt = 1)
        SMD_run_time_arr[rep_idx] = stat_SMD.total_solved_time
        FMD_run_time_arr[rep_idx] = stat_FMD.total_solved_time
        SMD_real_T_arr[rep_idx] = stat_SMD.real_T_list[0]
        FMD_real_T_arr[rep_idx] = stat_FMD.real_T_list[0]
        SMD_T_arr[rep_idx] = stat_SMD.max_iter
        FMD_T_arr[rep_idx] = stat_FMD.max_iter
        data_arr = np.vstack(
            (SMD_run_time_arr, SMD_real_T_arr, SMD_T_arr, FMD_run_time_arr, FMD_real_T_arr, FMD_T_arr)).T
        df = pd.DataFrame(data_arr, columns=['SMD_Solve_time', 'SMD_Iter', 'SMD_Max_Iter', \
                                             'FMD_Solve_time', 'FMD_Iter', 'FMD_Max_Iter'])

        custom_path = '/results/n_num_result/d={}/n={}_J={}_L={}_m={}_K={}.csv'.format(J_nt * L_nt, n_num, J_nt,
                                                                                       L_nt, m_nt, K_nt)
        save_path = parent_path + custom_path
        df.to_csv(save_path, index=False)
# temp_data_list.append(np.mean(SMD_run_time_arr))
# temp_data_list.append(np.mean(SMD_real_T_arr))
# temp_data_list.append(np.mean(SMD_T_arr))
# temp_data_list.append(np.mean(FMD_run_time_arr))
# temp_data_list.append(np.mean(FMD_real_T_arr))
# temp_data_list.append(np.mean(FMD_T_arr))
# data_list.append(temp_data_list)





