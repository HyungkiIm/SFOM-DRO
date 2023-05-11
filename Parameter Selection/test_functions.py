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
from SMD_Solver import *
from FMD_Solver import *
from utils import *


#Todo
"""
1. Need to fix K_test code.(Entirely)<-- Need to pay attention to this. 
2. Change x-axis of K_test plot to computation time.
"""


"""

This code includes two test functions: n_num_test and K_test.

"""

def n_num_test(n_num, J_nt, L_nt, m_nt,K_nt, print_opt,feas_opt):

    #Define Variables
    opt_low_nt = 0  # lower bound of optimal value
    opt_up_nt = 2  # upper bound of optimal value
    obj_tol_nt = 5e-2  # tolerance of our objective value\\ tolerance of R_x and R_i, obj_tol and alg_tol should be same.
    C_K_nt = 0.05  # proportion of epsilon that we need to pay by using i_hat
    alpha_tol_nt = 1e-10
    dual_gap_option_nt = 3
    dual_gap_freq_nt = 1/40
    var_scale_nt = 0.1  # Variance of our data
    # Uncertainty set parameter
    delta_nt = 0.9  # p \geq \delta/n
    rho_nt = 5  # D_f(p,q) \leq \rho/n
    # nu_nt = 0.1  # Prob of high convergence
    min_flag = 0
    T_cap_nt = 100000
    ss_type_nt = 'diminish'
    # Get statistics of various cases.
    #stat_list_nt[][][]
    stat_list_nt = []  # Save stat for figure 1, each element of this list contains one stat of alg for n_num1 simulation

    # Draw figure 1

    # Run the algorithms

    # alg_list should be [alg1,alg2,...,alg 10, ...] and for each alg_n it is consisted of [K,K_grad]
    # K could be either Sample size K that we are going to use or i_star
    # K_grad could be either SGD or FGD

    """

    Plot x-axis: n_num, y-axis: average time

    """

    K_grad_nt = 1

    # stat1_list[n_idx][repeat_idx][FMD, SMD]
    # Create our dataset
    emp_dist_value = np.zeros([m_nt, J_nt, L_nt, n_num])
    emp_dist_value[:] = np.nan
    x_0_nt = np.ones([J_nt, L_nt]) / L_nt
    p_0_nt = np.ones([m_nt, n_num]) / n_num

    #     emp_dist_value_1 = np.random.normal(loc=0.5, scale=0.125, size=(m_nt, J_nt, int(L_nt/2), n_num))
    #     emp_dist_value_2 = np.random.normal(loc=0.25, scale=0.125, size=(m_nt, J_nt, int((L_nt+1)/2), n_num))
    #     emp_dist_value = np.concatenate([emp_dist_value_1, emp_dist_value_2], axis=2)

    RHS_nt = np.zeros(m_nt)
    # Create random mean matrix for our data, [0,1] uniform distribution
    # We try to create | l | <= 0.5 instance.
    mean_array_nt = np.random.rand(m_nt, J_nt, L_nt) * 2 / J_nt
    for i in range(m_nt):
        RHS_nt[i] = np.sum(np.mean(mean_array_nt, axis=2)[i, :]) * 1.1

    print('Problem n = %s, m = %s, J = %s, L = %s' % (n_num, m_nt, J_nt, L_nt))

    # Random mean data generation
    emp_dist_value = np.zeros([m_nt, J_nt, L_nt, n_num])
    emp_dist_value[:] = np.nan
    for m_idx in range(m_nt):
        for j_idx in range(J_nt):
            for l_idx in range(L_nt):
                emp_dist_value[m_idx, j_idx, l_idx, :] = np.random.normal(
                    loc=mean_array_nt[m_idx, j_idx, l_idx], \
                    scale=var_scale_nt * mean_array_nt[m_idx, j_idx, l_idx], size=n_num)

    stoc_factor_nt = 1
    #Change this to alg_SMD
    #

    # Change data according to min_flag.

    stat_SMD = DRO_SMD(x_0_nt, p_0_nt, emp_dist_value, K_nt, K_grad_nt, delta_nt, rho_nt, alpha_tol_nt, \
                       opt_low_nt, opt_up_nt, obj_tol_nt, RHS_nt, ss_type_nt, C_K_nt,dual_gap_option_nt, dual_gap_freq_nt,
                       T_cap_nt, print_option=print_opt, K_test_flag=0, min_flag=min_flag, feas_opt = feas_opt)

    stat_FMD = DRO_FMD(x_0_nt, p_0_nt, emp_dist_value, delta_nt, rho_nt, alpha_tol_nt, opt_low_nt,\
                       opt_up_nt, obj_tol_nt,  RHS_nt,ss_type_nt,dual_gap_option_nt, dual_gap_freq_nt, T_cap_nt,\
                       print_option=print_opt, K_test_flag=0, min_flag=min_flag, feas_opt = feas_opt)

    return (stat_SMD, stat_FMD, n_num)

def K_test_iter(J_sst, L_sst, dual_gap_freq_sst, n_num, K_list_sst, T_cap_sst,
           m_sst, print_opt):

    #sst stands for sample size test.

    opt_low_sst = 0  # lower bound of optimal value
    opt_up_sst = 2  # upper bound of optimal value
    obj_tol_sst = 5e-2  # tolerance of our objective value\\ tolerance of R_x and R_i, obj_tol and alg_tol should be same.
    C_K_sst = 0.05  # proportion of epsilon that we need to pay by using i_hat
    alpha_tol_sst = 1e-10
    K_grad_sst = 1
    rho_sst = 5  # D_f(p,q) \leq \rho/n
    ss_type_sst = 'diminish'
    var_scale_sst = 0.2
    delta_sst = 0.9  # p \geq \delta/n
    dual_gap_option_sst = 2
    min_flag = 1


    mean_array_sst = np.random.rand(m_sst, J_sst, L_sst) / J_sst
    # Create our dataset
    emp_dist_value = np.zeros([m_sst, J_sst, L_sst, n_num])
    emp_dist_value[:] = np.nan
    x_0_sst = np.ones([J_sst, L_sst]) / L_sst
    p_0_sst = np.ones([m_sst, n_num]) / n_num
    RHS_sst = np.zeros(m_sst)
    for i in range(m_sst):
        RHS_sst[i] = np.sum(np.mean(mean_array_sst, axis=2)[i, :]) * 0.95
        # else:
        #     RHS_sst[i] = np.sum(np.mean(mean_array_sst, axis=2)[i, :]) * 1.1

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

    stat_list_sst = []

    for k_idx in range(len(K_list_sst)):
        SMD_stat = DRO_SMD_K_test_iter(x_0_sst, p_0_sst, emp_dist_value, K_list_sst[k_idx], K_grad_sst,
                              delta_sst, rho_sst, alpha_tol_sst, opt_low_sst, opt_up_sst,
                              obj_tol_sst, RHS_sst, ss_type_sst, C_K_sst, \
                              dual_gap_option_sst, dual_gap_freq_sst, T_cap_sst, print_option=print_opt, min_flag=min_flag)
        stat_list_sst.append(SMD_stat)
    FMD_stat = DRO_FMD_K_test_iter(x_0_sst, p_0_sst, emp_dist_value,delta_sst, rho_sst, alpha_tol_sst, opt_low_sst, opt_up_sst,
                              obj_tol_sst, RHS_sst, ss_type_sst,dual_gap_option_sst,\
                    dual_gap_freq_sst, T_cap_sst, print_option=print_opt, min_flag=min_flag)

    stat_list_sst.append(FMD_stat)

    return stat_list_sst

def K_test_time(J_sst, L_sst, n_sst, m_sst,emp_dist_value, RHS_sst, time_cap_sst, time_freq_sst,dual_gap_freq_sst, K_sst, print_opt):

    opt_low_sst = 0  # lower bound of optimal value
    opt_up_sst = 2  # upper bound of optimal value
    obj_tol_sst = 1e-3  # tolerance of our objective value\\ tolerance of R_x and R_i, obj_tol and alg_tol should be same.
    C_K_sst = 0  # proportion of epsilon that we need to pay by using i_hat
    alpha_tol_sst = 1e-10
    K_grad_sst = 1
    rho_sst = 5  # D_f(p,q) \leq \rho/n

    ss_type_sst = 'diminish'
    delta_sst = 0.9  # p \geq \delta/n
    dual_gap_option_sst = 2
    min_flag = 0

    x_0_sst = np.ones([J_sst, L_sst]) / L_sst
    p_0_sst = np.ones([m_sst, n_sst]) / n_sst


    if K_sst == n_sst:
        stat = DRO_FMD_K_test_time(x_0_sst, p_0_sst, emp_dist_value,
                                                        delta_sst, rho_sst, alpha_tol_sst, opt_low_sst, opt_up_sst,
                                                        obj_tol_sst, RHS_sst, ss_type_sst,
                                                        dual_gap_option_sst, dual_gap_freq_sst, time_cap_sst,time_freq_sst,
                                                        print_option=print_opt,
                                                        min_flag=min_flag)
    else:
        stat = DRO_SMD_K_test_time(x_0_sst, p_0_sst, emp_dist_value, K_sst, K_grad_sst,
                                       delta_sst, rho_sst, alpha_tol_sst, opt_low_sst, opt_up_sst,
                                       obj_tol_sst, RHS_sst, ss_type_sst, C_K_sst, \
                                       dual_gap_option_sst, dual_gap_freq_sst, time_cap_sst, time_freq_sst, print_option=print_opt,
                                       min_flag=min_flag)

    return stat


