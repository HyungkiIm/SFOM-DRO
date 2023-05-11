
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
from UBRegret import R_dim_SMD, R_const_SMD, R_dim_FMD, R_const_FMD
from utils import *

#Todo
"""
1. Go through Full_alpha function
2. Saving p value in tree structure?
3. For j in J for l in L improvement?
4. Split SMD and FMD
"""

"""
Update Notes

For previous update notes, please refer to v3.7.8.

Compare the computation time with v3.7.7. <-- need to do this
+ Changed Find_alpha function so that we don't need tolerance anymore.

v3.7.9.
Seperate our code to few different code. 
The third code is n_num and K_test function.

Oct-04
Added dual_flag feature. If dual_freq ==0, then it means turning off the dual calculation.
Oct-05 
Here we split SMD and FMD solver into two different functions.


"""








# Only used for K_test. Always K,K_grad<n. Return 1 if i_hat == i_*, return 0 otherwise.
# def SMD_A_x_K_test(K, K_grad, x, p, prox, divg, step_alpha, RHS, emp_dist_value, w_sum, cum_dist, C_K, obj_tol):
#     """
#     When Prox function is entropy function
#     If we use chi-square as a f-divergence, we need to relax our uncertainty set according to
#     Duchi's paper
#     """
#     # Get the value of m,J,L,n
#     m, J, L, n = emp_dist_value.shape
#     g_t = np.zeros([J, L])  # subgradient estimator
#     F_hat = []
#     F_star = []
#
#     """
#     i_hat calculation
#     """
#
#     K_samples = np.zeros([m, K])
#     samples_list = np.random.rand(m, K)
#
#     # Pick K samples from each constraints
#     for i in range(m):
#         K_samples[i, :] = np.digitize(random_samples[i, :], cum_dist[i, :])
#
#     K_samples = np.where(K_samples >= n, n - 1, K_samples)
#
#     """
#     Compute hat(F)^{i,K} for each constraint i
#     hat(F)^{i,K} = 1/K* sum_{j=1}^K F^i(x_t,z^{i,I_j^i})
#     """
#
#     for i in range(m):
#         temp = 0
#         for j in range(J):
#             for l in range(L):
#                 temp += np.sum(emp_dist_value[i, j, l, :][K_samples[i, :].astype(int)]) * x[j, l] * w_sum[i]
#         temp = temp / K - RHS[i]
#         F_hat.append(temp)
#
#     # get the index of max value in list F_hat
#     i_hat = F_hat.index(max(F_hat))
#
#     for i in range(m):
#         temp = 0
#         for j in range(J):
#             for l in range(L):
#                 temp += np.dot(emp_dist_value[i, j, l, :], p[i, :]) * x[j, l]
#         temp = temp - RHS[i]
#         F_star.append(temp)
#
#     i_star = F_star.index(max(F_star))
#
#     i_flag = 0
#     if abs(F_hat[i_hat] - F_star[i_hat]) < (1 - C_K) * obj_tol / 2:
#         i_flag = 1
#
#     """
#     g_t calculation
#     """
#
#     K_grad_samples = np.zeros(K_grad)
#     grad_samples_list = np.random.rand(K_grad)
#     K_grad_samples[:] = np.digitize(grad_samples_list, cum_dist[i_hat, :])
#     K_grad_samples = np.where(K_grad_samples >= n, n - 1, K_grad_samples)
#
#     for k_idx in range(K_grad):
#         g_t += w_sum[i_hat] * emp_dist_value[i_hat, :, :, int(K_grad_samples[k_idx])] / K_grad
#
#     """
#     x_{t+1} calculation
#     """
#
#     # Get next x (Projection Step with entropy prox function)
#     theta = np.zeros([J, L])
#     for j in range(J):
#         for l in range(L):
#             theta[j, l] = x[j, l] * np.exp(-step_alpha * g_t[j, l])
#
#     x_update = np.zeros([J, L])
#     for j in range(J):
#         for l in range(L):
#             x_update[j, l] = theta[j, l] / np.sum(theta[j, :])
#
#     return x_update, i_flag


# #First let's just use uniform distribution as x0.
# x_0 = np.ones([J,L])/L
# #x_0 = np.zeros([J,L])
# #x_0[:,0] = 1
# p_0 = np.ones([m,J,L,n])/n
# #Set random seed


# print(x_0)
# #print out x_1
# RHS = np.ones([m])
# RHS[0] = 5
# #SMD_A_x(K, x_0, p_0, 'entropy','chi-square', 0.01, RHS, emp_dist_value)

# Finding alpha such that g'(alpha) = 0
# w is a vector, n is our length of the vector w and delta and rho are given parameters.
# Tol is tolerance of alpha value


# In[9]:





# In[11]:





"""

Plot 2: different m

"""
#
# # Draw Figure 2
# # stat2_list[m_idx][n_idx][rep_idx][alg_idx]
# for m_num in m_list:
#     stat_m = []
#     for n_num in n_list2:
#         K = 500
#         stat_repeat = []
#
#         # Create our dataset
#         emp_dist_value = np.zeros([m_num, J, L, n_num])
#         emp_dist_value[:] = np.nan
#         p_0 = np.ones([m_num, n_num]) / n_num
#         RHS = np.ones([m_num])
#
#         emp_dist_value_1 = np.random.normal(loc=0.5, scale=0.125, size=(m_num, J, L - 1, n_num))
#         emp_dist_value_2 = np.random.normal(loc=0.25, scale=0.125, size=(m_num, J, 1, n_num))
#         emp_dist_value = np.concatenate([emp_dist_value_1, emp_dist_value_2], axis=2)
#         for rep_idx in range(repeats):
#             stat_temp = []
#
#             for idx in range(alg_num):
#                 alg_stat = DRO_Solver(alg_list[idx][0], alg_list[idx][1], x_0, p_0, emp_dist_value, nu, epsilon,
#                                       obj_tol, \
#                                       K, K_grad, 'entropy', 'chi-square', delta, rho, alpha_tol, opt_low, opt_up,
#                                       obj_tol, \
#                                       RHS, ss_type=ss_type1, min_flag=0)
#
#                 stat_temp.append(alg_stat)
#             stat_repeat.append(stat_temp)
#         stat_m.append(stat_m)
#     stat2_list.append(stat_repeat)
#
# total_runtime_list = []  # total_runtime_list[alg_idx][n_idx][m_idx]
#
# for alg_idx in range(alg_num):
#     alg_time_list = []
#     for n_idx in range(len(n_list2)):
#         n_time_list = []
#         for m_idx in range(len(m_list)):
#             temp = 0
#             for rep_idx in range(repeats):
#                 temp += stat2_list[m_idx][n_idx][rep_idx][alg_idx].total_solved_time
#             temp /= repeats
#             n_time_list.append(temp)
#         alg_time_list.append(n_time_list)
#     total_runtime_list.append(alg_time_list)
#
# # total_runtime_list
#
# plt.figure(figsize=(16, 10))
#
# for n_idx in range(len(n_list2)):
#     plt.subplot(2, math.ceil(len(n_list2) / 2), n_idx + 1)
#     for alg_idx in range(alg_num):
#         plt.plot(m_list, total_runtime_list[alg_idx][n_idx], c=color_list[alg_idx], ls=ls_list[alg_idx], \
#                  marker=marker_list[alg_idx], label=alg_name[alg_idx])
#         plt.xlabel('m')
#         plt.ylabel('time(s)')
#         plt.title('n=%s' % n_list2[n_idx])
#         plt.legend(loc="upper left")
#
# plt.show()


# In[19]:



