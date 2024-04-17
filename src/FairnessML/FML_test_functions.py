import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import gurobipy as gp
from gurobipy import GRB
from statistics import mean
from tqdm import tqdm
from FML_SMD_Solver import DRO_SMD, DRO_SMD_K_test_time
from FML_FMD_Solver import DRO_FMD, DRO_FMD_K_test_time
from copy import deepcopy


"""

Risk Averse Example

"""

def n_num_test(n_num,K_nt, X_train, y_train, c, print_opt,feas_opt,warm_start):

    """
    
    This function tests the performance (solving time) of SMD and FMD for different number of samples (n).

    """

    opt_low_nt = 0  # lower bound of optimal value
    opt_up_nt = 1 # upper bound of optimal value
    obj_tol_nt = 2e-2  # tolerance of our objective value\\ tolerance of R_x and R_i, obj_tol and alg_tol should be same.
    C_K_nt = 0.05  # proportion of epsilon that we need to pay by using i_hat
    alpha_tol_nt = 1e-10
    K_grad_nt = 200
    rho_nt = 5  # D_f(p,q) \leq \rho/n
    ss_type_nt = 'diminish' # step size type
    delta_nt = 0.95  # p \geq \delta/n
    dual_gap_option_nt = 1
    dual_gap_freq_nt_SMD = 1/20
    dual_gap_freq_nt2_SMD = 1/20
    dual_gap_freq_nt_FMD = 1/20
    dual_gap_freq_nt2_FMD = 1/20
    min_flag = 1 # 1 if our problem is minimization problem
    T_cap_nt = 10000000 # Max_iter

    #Initialize
    m = 3
    stoc_factor_nt = 1
    n, d = X_train.shape
    x_0_nt = np.zeros(d)
    p_0_nt = np.ones((m,n_num)) / n_num
    RHS_nt = np.zeros(3)
    RHS_nt[1:] = c

    #Perform SMD
    stat_SMD = DRO_SMD(x_0_nt, p_0_nt, X_train, y_train, K_nt, \
                       K_grad_nt, delta_nt, rho_nt, alpha_tol_nt, \
                       opt_low_nt, opt_up_nt, obj_tol_nt, RHS_nt, ss_type_nt, C_K_nt,dual_gap_option_nt, dual_gap_freq_nt_SMD,dual_gap_freq_nt2_SMD,
                        print_option=print_opt, K_test_flag=0, min_flag=min_flag, feas_opt = feas_opt, warm_start= warm_start)

    #Perform FMD
    stat_FMD = DRO_FMD(x_0_nt, p_0_nt, X_train,y_train, delta_nt, rho_nt, alpha_tol_nt, opt_low_nt,\
                       opt_up_nt, obj_tol_nt,  RHS_nt,ss_type_nt,dual_gap_option_nt, dual_gap_freq_nt_FMD,dual_gap_freq_nt2_FMD,
                       print_option=print_opt, K_test_flag=0, min_flag=min_flag, feas_opt = feas_opt, warm_start = warm_start)

    return (stat_SMD, stat_FMD, n_num)

def K_test_time(n_sst, X_train, y_train, c, time_cap_sst, time_freq_sst,dual_gap_freq_sst, K_sst, print_opt):

    opt_low_sst = 0  # lower bound of optimal value
    opt_up_sst = 1  # upper bound of optimal value
    obj_tol_sst = 1e-3  # tolerance of our objective value\\ tolerance of R_x and R_i, obj_tol and alg_tol should be same.
    C_K_sst = 0.05  # proportion of epsilon that we need to pay by using i_hat
    alpha_tol_sst = 1e-10
    K_grad_sst = 200
    rho_sst = 5  # D_f(p,q) \leq \rho/n

    ss_type_sst = 'diminish'
    delta_sst = 0.95  # p \geq \delta/n
    dual_gap_option_sst = 2
    min_flag = 1

    #Initialize
    m = 3
    stoc_factor_nt = 1
    n, d = X_train.shape
    x_0_sst = np.zeros(d)
    p_0_sst = np.ones((m,n_sst)) / n_sst
    RHS_sst = np.zeros(3)
    RHS_sst[1:] = c


    if K_sst == n_sst:
        stat = DRO_FMD_K_test_time(x_0_sst, p_0_sst, X_train, y_train,
                                                        delta_sst, rho_sst, alpha_tol_sst, opt_low_sst, opt_up_sst,
                                                        obj_tol_sst, RHS_sst, ss_type_sst,
                                                        dual_gap_option_sst, dual_gap_freq_sst, time_cap_sst,time_freq_sst,
                                                        print_option=print_opt,
                                                        min_flag=min_flag)
    else:
        stat = DRO_SMD_K_test_time(x_0_sst, p_0_sst, X_train, y_train, K_sst, K_grad_sst,
                                       delta_sst, rho_sst, alpha_tol_sst, opt_low_sst, opt_up_sst,
                                       obj_tol_sst, RHS_sst, ss_type_sst, C_K_sst, \
                                       dual_gap_option_sst, dual_gap_freq_sst, time_cap_sst, time_freq_sst, print_option=print_opt,
                                       min_flag=min_flag)

    return stat


