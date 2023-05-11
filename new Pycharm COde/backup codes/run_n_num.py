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

Here we plot the computation time of SMD and FMd for different n values. 

Update Notes: 
Fixed plotting codes.


"""

J_nt = 5  # Number of Cohorts
L_nt = 5 # Number of Treatments

opt_low_nt = 0  # lower bound of optimal value
opt_up_nt = 2  # upper bound of optimal value
obj_tol_nt = 1e-1  # tolerance of our objective value\\ tolerance of R_x and R_i, obj_tol and alg_tol should be same.
C_K_nt = 0.95  # proportion of epsilon that we need to pay by using i_hat
alpha_tol_nt = 1e-10
dual_gap_freq_nt = 0
var_scale_nt = 0.05  # Variance of our data
# m = 3  # number of constraints, first index 0 refers to our objective function.
# K = 100  # Sample size of our gradient estimator
# K_grad = 1
# N = 100  # size of each uncertainty set\\ We are no longer using this parameter in this code
# n = 10 ** 2  # sample size of simulation data set, also support size of each uncertainty set.

# nu = 0.1 # |g_t - h_t| <= nu
# epsilon = 0.001 #epsilon-subgradient


# Uncertainty set parameter
delta_nt = 0.8  # p \geq \delta/n
rho_nt = 5 # D_f(p,q) \leq \rho/n
#nu_nt = 0.1  # Prob of high convergence
seed_nt = 1234
min_flag = 0

# emp_dist_value_1 = np.random.normal(loc=0.5, scale=0.125, size=(m, J, L - 1, n))
# emp_dist_value_2 = np.random.normal(loc=0.25, scale=0.125, size=(m, J, 1, n))
# emp_dist_value = np.concatenate([emp_dist_value_1, emp_dist_value_2], axis=2)
# RHS = np.ones([m])

# Get statistics of various cases.

# n_list_nt = np.arange(1000,5500,500)
n_list_nt = [1000,1500]
m_nt = 5  # constraint number for figure 1
T_cap_nt = 100000

repeats_nt = 1

ss_type_nt = 'diminish'

# Run the algorithms

# alg_list should be [alg1,alg2,...,alg 10, ...] and for each alg_n it is consisted of [K,K_grad]
# K could be either Sample size K that we are going to use or i_star
# K_grad could be either SGD or FGD
#alg_list_nt = [['i_star', 'FGD'], [100, 1]]
K_list_nt = [100, 1]

stat_list_nt_diminish = n_num_test(n_list_nt, J_nt, L_nt, opt_low_nt, opt_up_nt, obj_tol_nt, C_K_nt, \
                                   alpha_tol_nt, dual_gap_freq_nt,T_cap_nt, var_scale_nt, delta_nt, rho_nt, seed_nt, \
                                   m_nt, repeats_nt, ss_type_nt, K_list_nt, min_flag, print_opt=1)

"""

Plot x-axis: n_num, y-axis: average time

"""

"""
#Create random mean matrix for our data, [0,0.25] uniform distribution
mean_array_sst = np.random.rand(m_nt, J_nt,L_nt) / J_nt

# stat1_list[n_idx][repeat_idx][alg_idx]
for n_num in n_list_nt:

    stat_repeat = []

    # Create our dataset
    emp_dist_value = np.zeros([m_nt, J_nt, L_nt, n_num])
    emp_dist_value[:] = np.nan
    x_0_nt = np.ones([J_nt, L_nt]) / L_nt
    p_0_nt = np.ones([m_nt, n_num]) / n_num
    RHS_nt = np.ones([m_nt])




#     emp_dist_value_1 = np.random.normal(loc=0.5, scale=0.125, size=(m_nt, J_nt, int(L_nt/2), n_num))
#     emp_dist_value_2 = np.random.normal(loc=0.25, scale=0.125, size=(m_nt, J_nt, int((L_nt+1)/2), n_num))
#     emp_dist_value = np.concatenate([emp_dist_value_1, emp_dist_value_2], axis=2)

    for rep_idx in tqdm(range(repeats_nt)):

        print('Problem n = %s, m = %s, J = %s, L = %s' %(n_num, m_nt,J_nt,L_nt))

        #Random mean data generation
        emp_dist_value = np.zeros([m_nt,J_nt,L_nt,n_num])
        emp_dist_value[:] = np.nan
        for m_idx in range(m_nt):
            for j_idx in range(J_nt):
                for l_idx in range(L_nt):
                    emp_dist_value[m_idx,j_idx,l_idx,:] = np.random.normal(loc = mean_array_sst[m_idx,j_idx, l_idx],\
                         scale= var_scale_nt * mean_array_sst[m_idx,j_idx, l_idx], size = n_num)




        G_nt = np.absolute(emp_dist_value).max()
        print('G:', G_nt)
        M_nt = []
        for i in range(m_nt):
            M_nt.append(J_nt * np.absolute(emp_dist_value[i, :, :, :]).max())



        stat_alg = []

        for alg_idx in range(len(alg_list_nt)):
            if alg_list_nt[alg_idx][0] == 'i_star':
                K_nt = n_num
            else:
                K_nt = alg_list_nt[alg_idx][0]

            if alg_list_nt[alg_idx][1] == 'FGD':
                K_grad_nt = n_num
            else:
                K_grad_nt = alg_list_nt[alg_idx][1]

            if K_nt == n_num:
                C_K_nt = 1

            # Calculate stoc_factor if K_grad < n

            # Omega = np.log(2 * m_1 / nu)
            #
            # if K_grad < n_num:
            #     stoc_factor = 1 + Omega / 2 + 4 * math.sqrt(Omega)

            stoc_factor_nt = 1

            if ss_type_nt == 'constant':
                T_nt, R_x, R_p, ss_x, ss_p = R_x_p_combined_constant(J_nt, L_nt, n_num, G_nt, M_nt, delta_nt, rho_nt,\
                                                        obj_tol_nt,C_K_nt, stoc_factor_nt, K_grad_nt)
            elif ss_type_nt == 'diminish':
                T_nt, R_x, R_p, ss_x, ss_p = R_x_p_combined_diminish(J_nt, L_nt, n_num, G_nt, M_nt, delta_nt, rho_nt,\
                                                        obj_tol_nt,C_K_nt, stoc_factor_nt, K_grad_nt)
            print("Max Iteration:", T_nt)
            # Generate random_samples_list
            random_samples_list = np.random.rand(T_nt, m_nt, K_nt)

            alg_stat = DRO_Solver(x_0_nt, p_0_nt, emp_dist_value,\
                                      K_nt, K_grad_nt, 'entropy', 'chi-square', delta_nt, rho_nt, nu_nt, alpha_tol_nt,\
                                      opt_low_nt, opt_up_nt, obj_tol_nt,
                                      RHS_nt, ss_type_nt, \
                                      random_samples_list, C_K_nt, dual_gap_freq_nt,T_cap_nt,\
                                      print_option=1, K_test_flag= 0, min_flag=0)

            stat_alg.append(alg_stat)
        stat_repeat.append(stat_alg)
    stat_list_nt.append(stat_repeat)

if len(alg_list_nt) == 2:
    marker_list = ['o', 'd']
    color_list = ['b', 'g']
    ls_list = ['-', '--']  # line Style
    alg_name = ['Stoch Subgrad', 'Full Subgrad']
"""

# Draw Plots

# Change the order of the list first
# Also calculate average solved time
avg_runtime_list_nt = []  # total_run_time_list[alg_idx][n_idx]

# In[ ]:


for alg_idx in range(2):
    temp_list = []

    for n_idx in range(len(n_list_nt)):
        temp = 0

        for rep_idx in range(repeats_nt):
            temp += stat_list_nt_diminish[n_idx][rep_idx][alg_idx].total_solved_time
        temp /= repeats_nt
        temp_list.append(temp)

    avg_runtime_list_nt.append(temp_list)

alg_name = ['FMD Dim', 'SMD Dim']

for alg_idx in range(len(avg_runtime_list_nt)):
    plt.plot(n_list_nt, avg_runtime_list_nt[alg_idx], label=alg_name[alg_idx])
    plt.xlabel('n')
    plt.ylabel('time(s)')
    plt.title('Average Solve Times for Different n (m=%s)' % m_nt)
    plt.legend(loc="upper left")

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
save_path = parent_path + '/results/diff_n_epsilon_' + str(obj_tol_nt) + '.png'

plt.savefig(save_path)

plt.show()