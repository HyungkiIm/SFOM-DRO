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
from PS_SMD_Solver import DRO_SMD
from PS_FMD_Solver import DRO_FMD
from PS_utils import *

def n_num_test(n_num, J_nt, L_nt, m_nt,K_nt, print_opt,feas_opt):

    #Define Variables
    opt_low_nt = 0  # lower bound of optimal value
    opt_up_nt = 2  # upper bound of optimal value
    obj_tol_nt = 5e-2  # tolerance of our objective value\\ tolerance of R_x and R_i, obj_tol and alg_tol should be same.
    C_K_nt = 0.05  # proportion of epsilon that we need to pay by using i_hat
    alpha_tol_nt = 1e-10
    dual_gap_option_nt = 3
    dual_gap_freq_nt = 1/20
    var_scale_nt = 0.1  # Variance of our data
    # Uncertainty set parameter
    delta_nt = 0.9  # p \geq \delta/n
    rho_nt = 5  # D_f(p,q) \leq \rho/n
    # nu_nt = 0.1  # Prob of high convergence
    min_flag = 0
    T_cap_nt = 100000
    ss_type_nt = 'diminish'
    # Get statistics of various cases.
    stat_list_nt = []  # Save stat for figure 1, each element of this list contains one stat of alg for n_num1 simulation

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

    stat_SMD = DRO_SMD(x_0_nt, p_0_nt, emp_dist_value, K_nt, K_grad_nt, delta_nt, rho_nt, alpha_tol_nt, \
                       opt_low_nt, opt_up_nt, obj_tol_nt, RHS_nt, ss_type_nt, C_K_nt,dual_gap_option_nt, dual_gap_freq_nt,
                       T_cap_nt, print_option=print_opt, K_test_flag=0, min_flag=min_flag, feas_opt = feas_opt)

    stat_FMD = DRO_FMD(x_0_nt, p_0_nt, emp_dist_value, delta_nt, rho_nt, alpha_tol_nt, opt_low_nt,\
                       opt_up_nt, obj_tol_nt,  RHS_nt,ss_type_nt,dual_gap_option_nt, dual_gap_freq_nt, T_cap_nt,\
                       print_option=print_opt, K_test_flag=0, min_flag=min_flag, feas_opt = feas_opt)

    return (stat_SMD, stat_FMD, n_num)



