import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from FML_test_functions import n_num_test


"""
Risk-Averse Example
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
n_list_nt = [8000]

seed_nt = 12345
np.random.seed(seed_nt)
#n_list_nt = np.linspace(10000,20000,11)
c = 0.05 #Second and Third Constraint RHS

repeats_nt = 1
K_nt = 200
print_opt = 1
warm_start = 0
feas_opt =  1

SMD_run_time_arr = np.zeros(repeats_nt)
FMD_run_time_arr = np.zeros(repeats_nt)
SMD_real_T_arr = np.zeros(repeats_nt)
FMD_real_T_arr = np.zeros(repeats_nt)
SMD_T_arr = np.zeros(repeats_nt)
FMD_T_arr = np.zeros(repeats_nt)
SMD_solved = np.zeros(repeats_nt)
FMD_solved = np.zeros(repeats_nt)
SMD_dualgap = np.zeros(repeats_nt)
FMD_dualgap = np.zeros(repeats_nt)
SMD_dualgap_time = np.zeros(repeats_nt)
FMD_dualgap_time = np.zeros(repeats_nt)

for n_num in n_list_nt:
    for rep_idx in range(repeats_nt):
        #Create dataset randomly
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= n_num / n_samples, shuffle=True, random_state=77)
        stat_SMD, stat_FMD, _ = n_num_test(int(n_num), K_nt, X_train, y_train, c, print_opt = print_opt, feas_opt = feas_opt, warm_start=warm_start)
        SMD_run_time_arr[rep_idx] = stat_SMD.total_solved_time
        FMD_run_time_arr[rep_idx] = stat_FMD.total_solved_time
        SMD_real_T_arr[rep_idx] = stat_SMD.real_T_list[0]
        FMD_real_T_arr[rep_idx] = stat_FMD.real_T_list[0]
        SMD_T_arr[rep_idx] = stat_SMD.max_iter
        FMD_T_arr[rep_idx] = stat_FMD.max_iter
        SMD_solved[rep_idx] = stat_SMD.solved
        FMD_solved[rep_idx] = stat_FMD.solved
        SMD_dualgap[rep_idx] = stat_SMD.dual_gap_list[0][0]
        FMD_dualgap[rep_idx] = stat_FMD.dual_gap_list[0][0]
        SMD_solved[rep_idx] = stat_SMD.solved
        FMD_solved[rep_idx] = stat_FMD.solved
        SMD_dualgap_time[rep_idx] = stat_SMD.dual_gap_time
        FMD_dualgap_time[rep_idx] = stat_FMD.dual_gap_time

    data_arr = np.vstack(
        (SMD_run_time_arr, SMD_real_T_arr, SMD_T_arr, SMD_solved, SMD_dualgap, SMD_dualgap_time,\
         FMD_run_time_arr, FMD_real_T_arr, FMD_T_arr, FMD_solved, FMD_dualgap, FMD_dualgap_time)).T
    df = pd.DataFrame(data_arr, columns=['SMD_Solve_time', 'SMD_Iter', 'SMD_Max_Iter', 'SMD_solved', 'SMD_gap', 'SMD_dual_time', \
                                         'FMD_Solve_time', 'FMD_Iter', 'FMD_Max_Iter','FMD_solved', 'FMD_gap', 'FMD_dual_time'])

    custom_path = '/results/n_num_result/n={}_K={}_poly={}.csv'.format(n_num, K_nt,poly_degree)
    save_path = parent_path + custom_path
    df.to_csv(save_path, index=False)


# temp_data_list.append(np.mean(SMD_run_time_arr))
# temp_data_list.append(np.mean(SMD_real_T_arr))
# temp_data_list.append(np.mean(SMD_T_arr))
# temp_data_list.append(np.mean(FMD_run_time_arr))
# temp_data_list.append(np.mean(FMD_real_T_arr))
# temp_data_list.append(np.mean(FMD_T_arr))
# data_list.append(temp_data_list)
# data_arr = np.vstack((SMD_run_time_arr,SMD_real_T_arr,SMD_T_arr,FMD_run_time_arr,FMD_real_T_arr,FMD_T_arr)).T
#
#
# df= pd.DataFrame(data_arr, columns = ['SMD_Solve_time', 'SMD_Iter', 'SMD_Max_Iter',\
#                 'FMD_Solve_time', 'FMD_Iter', 'FMD_Max_Iter'])
#
# custom_path = '/results/n_num_result/d={}/n={}_J={}_L={}_m={}_K={}.csv'.format(J_nt*L_nt,n_list_nt[0],J_nt,L_nt,m_nt,K_nt)
# save_path = parent_path + custom_path
# df.to_csv(save_path, index=False)

