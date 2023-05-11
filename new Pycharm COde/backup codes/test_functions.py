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
from SMD_Solver import DRO_SMD
from FMD_Solver import DRO_FMD


#Todo
"""
1. Need to fix K_test code.(Entirely)<-- Need to pay attention to this. 
2. Change x-axis of K_test plot to computation time.
"""


"""

This code includes two test functions: n_num_test and K_test.

"""

def n_num_test(n_list_nt, J_nt, L_nt, opt_low_nt, opt_up_nt, obj_tol_nt, C_K_nt, alpha_tol_nt,
               dual_gap_freq_nt,T_cap_nt, var_scale_nt, delta_nt, rho_nt, seed_nt, m_nt, repeats_nt, ss_type_nt,
               K__list_nt, min_flag, print_opt):
    np.random.seed(seed_nt)
    # emp_dist_value_1 = np.random.normal(loc=0.5, scale=0.125, size=(m, J, L - 1, n))
    # emp_dist_value_2 = np.random.normal(loc=0.25, scale=0.125, size=(m, J, 1, n))
    # emp_dist_value = np.concatenate([emp_dist_value_1, emp_dist_value_2], axis=2)
    # RHS = np.ones([m])

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

    K_nt = K__list_nt[0]
    K_grad_nt = K__list_nt[1]

    # stat1_list[n_idx][repeat_idx][FMD, SMD]
    for n_num in n_list_nt:

        stat_repeat = []

        # Create our dataset
        emp_dist_value = np.zeros([m_nt, J_nt, L_nt, n_num])
        emp_dist_value[:] = np.nan
        x_0_nt = np.ones([J_nt, L_nt]) / L_nt
        p_0_nt = np.ones([m_nt, n_num]) / n_num

        #     emp_dist_value_1 = np.random.normal(loc=0.5, scale=0.125, size=(m_nt, J_nt, int(L_nt/2), n_num))
        #     emp_dist_value_2 = np.random.normal(loc=0.25, scale=0.125, size=(m_nt, J_nt, int((L_nt+1)/2), n_num))
        #     emp_dist_value = np.concatenate([emp_dist_value_1, emp_dist_value_2], axis=2)

        for rep_idx in tqdm(range(repeats_nt)):
            RHS_nt = np.zeros(m_nt)
            # Create random mean matrix for our data, [0,1] uniform distribution
            # We try to create | l | <= 0.5 instance.
            mean_array_sst = np.random.rand(m_nt, J_nt, L_nt) * 2 / J_nt
            for i in range(m_nt):
                RHS_nt[i] = np.sum(np.mean(mean_array_sst, axis=2)[i, :]) * 1.1

            print('Problem n = %s, m = %s, J = %s, L = %s' % (n_num, m_nt, J_nt, L_nt))

            # Random mean data generation
            emp_dist_value = np.zeros([m_nt, J_nt, L_nt, n_num])
            emp_dist_value[:] = np.nan
            for m_idx in range(m_nt):
                for j_idx in range(J_nt):
                    for l_idx in range(L_nt):
                        emp_dist_value[m_idx, j_idx, l_idx, :] = np.random.normal(
                            loc=mean_array_sst[m_idx, j_idx, l_idx], \
                            scale=var_scale_nt * mean_array_sst[m_idx, j_idx, l_idx], size=n_num)

            stat_alg = []


            stoc_factor_nt = 1
            #Change this to alg_SMD
            #
            stat_SMD = DRO_SMD(x_0_nt, p_0_nt, emp_dist_value, K_nt, K_grad_nt, delta_nt, rho_nt, alpha_tol_nt, \
                               opt_low_nt, opt_up_nt, obj_tol_nt, RHS_nt, ss_type_nt, C_K_nt, dual_gap_freq_nt,
                               T_cap_nt, print_option=print_opt, K_test_flag=0, min_flag=min_flag)
            stat_alg.append(stat_SMD)

            stat_FMD = DRO_FMD(x_0_nt, p_0_nt, emp_dist_value, delta_nt, rho_nt, alpha_tol_nt, opt_low_nt,\
                               opt_up_nt, obj_tol_nt,  RHS_nt,ss_type_nt, dual_gap_freq_nt, T_cap_nt,\
                               print_option=print_opt, K_test_flag=0, min_flag=min_flag)
            stat_alg.append(stat_FMD)

            stat_SMD = DRO_SMD(x_0_nt, p_0_nt, emp_dist_value, K_nt, K_grad_nt, delta_nt, rho_nt, alpha_tol_nt,\
                               opt_low_nt, opt_up_nt, obj_tol_nt, RHS_nt, ss_type_nt, C_K_nt, dual_gap_freq_nt,
                               T_cap_nt, print_option=print_opt, K_test_flag=0, min_flag=min_flag)
            stat_alg.append(stat_SMD)

            stat_repeat.append(stat_alg)
        stat_list_nt.append(stat_repeat)

    return stat_list_nt


def K_test(J_sst, L_sst, opt_low_sst, obj_tol_sst, C_K_sst, alpha_tol_sst, dual_gap_freq_sst,
           delta_sst, rho_sst, nu_sst, n_list_sst, K_list_sst, K_grad_sst, T_cap_sst, ss_type_sst, repeats_sst,
           m_sst, var_scale_sst, print_opt):
    alg_list_sst = []
    for k_idx in range(len(K_list_sst)):
        temp = []
        temp.append(K_list_sst[k_idx])
        temp.append(K_grad_sst)
        alg_list_sst.append(temp)

    stat_list_sst = []

    for n_num in n_list_sst:
        stat_repeat = []

        for rep_idx in tqdm(range(repeats_sst)):
            stat_alg = []

            mean_array_sst = np.random.rand(m_sst, J_sst, L_sst) / J_sst
            # Create our dataset
            emp_dist_value = np.zeros([m_sst, J_sst, L_sst, n_num])
            emp_dist_value[:] = np.nan
            x_0_sst = np.ones([J_sst, L_sst]) / L_sst
            p_0_sst = np.ones([m_sst, n_num]) / n_num
            RHS_sst = 0.4 * np.ones([m_sst])  # RHS is important for determining K!

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

            # Get Approximated Obj Val
            approx_obj_sst = 0
            for j_idx in range(J_sst):
                approx_obj_sst += np.max(mean_array_sst[0, j_idx, :])

            print('approx_obj_sst:', approx_obj_sst)
            opt_up_sst = 2 * approx_obj_sst

            G_sst = np.absolute(emp_dist_value).max()
            print('G:', G_sst)
            M_sst = []
            for i in range(m_sst):
                M_sst.append(J_sst * np.absolute(emp_dist_value[i, :, :, :]).max())

            for alg_idx in range(len(alg_list_sst)):

                if alg_list_sst[alg_idx][0] == 'i_star':
                    K_sst = n_num
                else:
                    K_sst = alg_list_sst[alg_idx][0]

                if alg_list_sst[alg_idx][1] == 'FGD':
                    K_grad_sst = n_num
                else:
                    K_grad_sst = alg_list_sst[alg_idx][1]

                if K_sst == n_num:
                    C_K_sst = 1

                random_samples_list = np.random.rand(T_cap_sst, m_sst, K_sst)

                # Repeat on the same dataset we have

                print("K_sst:", K_sst)
                alg_stat = DRO_SMD(x_0_sst, p_0_sst, emp_dist_value, K_sst, K_grad_sst, 'entropy', 'chi-square',
                                      delta_sst, rho_sst, nu_sst, alpha_tol_sst, opt_low_sst, opt_up_sst,
                                      obj_tol_sst, RHS_sst, ss_type_sst, random_samples_list, C_K_sst, \
                                      dual_gap_freq_sst, T_cap_sst, print_option=0, K_test_flag=1, min_flag=0)

                stat_alg.append(alg_stat)
            stat_repeat.append(stat_repeat)
        stat_list_sst.append(stat_alg)

    return stat_list_sst




