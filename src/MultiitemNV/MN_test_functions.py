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
from MN_SMD_Solver import DRO_SMD
from MN_FMD_Solver import DRO_FMD
from MN_utils import get_L
from copy import deepcopy

def n_num_test(n_num, d_nt, beta_nt, K_nt, print_opt,feas_opt,warm_start):

    opt_low_nt = -1  # lower bound of optimal value
    opt_up_nt = 1  # upper bound of optimal value
    obj_tol_nt = 3e-2  # tolerance of our objective value\\ tolerance of R_x and R_i, obj_tol and alg_tol should be same.
    C_K_nt = 0.05  # proportion of epsilon that we need to pay by using i_hat
    alpha_tol_nt = 1e-10
    K_grad_nt = 1
    rho_nt = 5  # D_f(p,q) \leq \rho/n
    ss_type_nt = 'diminish'
    delta_nt = 0.9  # p \geq \delta/n
    dual_gap_option_nt = 3
    dual_gap_freq_nt_SMD = 1/5
    dual_gap_freq_nt2_SMD = 1/10
    dual_gap_freq_nt_FMD = 1 / 5
    dual_gap_freq_nt2_FMD = 1 / 20
    min_flag = 1
    T_cap_nt = 10000000

    d1 = 0.1
    d2 = 0.2
    r_vec = 0.5 * np.ones(d_nt)  # v
    c_vec = np.random.uniform(low=0.1, high=0.25, size=d_nt)  # c
    s_vec = 0.2 * r_vec  # g
    b_vec = 0.25 * r_vec  # b

    mean_vec = np.random.uniform(low = d1, high = d2, size = d_nt)
    sd_vec = np.random.uniform(low = 0.05 * mean_vec, high = 0.2 * mean_vec)
    #Generate Cov Matrix
    S = np.random.normal(size = (d_nt,d_nt))
    U = S.T @ S
    u = 1 / np.sqrt(np.diag(U))
    C = np.diag(u) @ U @ np.diag(u)
    Cov_mat = np.diag(sd_vec) @ C @ np.diag(sd_vec)
    xi = np.random.multivariate_normal(mean_vec, Cov_mat, size = n_num)
    emp_dist_value = deepcopy(xi.T)
    print('Problem n = %s, d = %s' % (n_num, d_nt))

    #First element of x_0_nt is tau
    x_0_nt = (d1+d2) / 2 * np.ones(d_nt+1)
    p_0_nt = np.ones((2,n_num)) / n_num
    Loss_vec = get_L(c_vec, s_vec, r_vec, b_vec,x_0_nt[1:],emp_dist_value)
    x_0_nt[0] = np.quantile(Loss_vec, 1 - beta_nt)
    budget_nt = 1.2 * np.sum(mean_vec)
    #x_bar equals mean_vec
    Loss_vec_bar = get_L(c_vec, s_vec, r_vec, b_vec, mean_vec,emp_dist_value)
    tau_bar = np.quantile(Loss_vec_bar, 1- beta_nt)
    alpha = np.average(tau_bar + 1/ beta_nt * (np.where(Loss_vec_bar > tau_bar,Loss_vec_bar - tau_bar, 0)))
    RHS_nt = np.array([0, alpha])
    stoc_factor_nt = 1
    #Change this to alg_SMD


    # Change data according to min_flag.
    stat_SMD = DRO_SMD(c_vec,s_vec, r_vec, b_vec, mean_vec, beta_nt, budget_nt, x_0_nt, p_0_nt, emp_dist_value, K_nt, \
                       K_grad_nt, delta_nt, rho_nt, alpha_tol_nt, \
                       opt_low_nt, opt_up_nt, obj_tol_nt, RHS_nt, ss_type_nt, C_K_nt,dual_gap_option_nt, dual_gap_freq_nt_SMD,dual_gap_freq_nt2_SMD,
                        print_option=print_opt, K_test_flag=0, min_flag=min_flag, feas_opt = feas_opt, warm_start= warm_start)

    stat_FMD = DRO_FMD(c_vec,s_vec, r_vec, b_vec, mean_vec, beta_nt, budget_nt,x_0_nt, p_0_nt, emp_dist_value, delta_nt, rho_nt, alpha_tol_nt, opt_low_nt,\
                       opt_up_nt, obj_tol_nt,  RHS_nt,ss_type_nt,dual_gap_option_nt, dual_gap_freq_nt_FMD,dual_gap_freq_nt2_FMD,
                       print_option=print_opt, K_test_flag=0, min_flag=min_flag, feas_opt = feas_opt, warm_start = warm_start)

    return (stat_SMD, stat_FMD, n_num)
