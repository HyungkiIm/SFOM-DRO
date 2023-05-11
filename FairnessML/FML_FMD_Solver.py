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
from FML_UBRegret import R_dim_FMD
from FML_utils import *
from copy import deepcopy
import torch


def FMD_x(x, p,step_alpha, X_train, y_train, s_norm, RHS):
    """
    Full gradient descent on x
    """
    n, d = X_train.shape
    m = 3
    F_hat = []

    """
    i_hat calculation
    """
    p_sum = np.sum(p, axis = 1)
    X_theta = X_train @ x
    #max_list = []
    for i in range(m):
        if i == 0:
            f_val = p[i,:] @ (np.log(1 + np.exp(X_theta)) - np.multiply(1-y_train, X_theta))
            f_val -= RHS[i] * p_sum[i]
            F_hat.append(f_val)
        elif i == 1:
            f_val = p[i,:] @ np.multiply(s_norm, X_theta)
            f_val -= RHS[i] * p_sum[i]
            F_hat.append(f_val)
        else:
            f_val = p[i, :] @ np.multiply(-s_norm, X_theta)
            f_val -= RHS[i] * p_sum[i]
            F_hat.append(f_val)

    i_hat = F_hat.index(max(F_hat))

    """
    g_t calculation
    """
    if i_hat == 0:
        temp = np.exp(X_theta)
        g_t = X_train.T @ (np.multiply(p[i_hat,:], temp /(1+ temp) - 1 + y_train))
    elif i_hat == 1:
        g_t = X_train.T @ (np.multiply(p[i_hat,:],s_norm))
    else:
        g_t = X_train.T @ (np.multiply(p[i_hat, :],-s_norm))

    """
    x_{t+1} calculation
    """
    x_update = x - step_alpha * g_t

    return x_update, np.array(F_hat)


def find_alpha_full(w, n, delta, rho, tol):
    alpha_up = 1
    alpha_low = 0
    g_alpha = 1
    esp = 1e-10
    # First sort w vector
    while abs(g_alpha) > tol:

        alpha = (alpha_up + alpha_low) / 2

        if alpha <= esp or 1 - alpha <= esp:
            return alpha

        w_threshold = (delta - alpha) / (n * (1 - alpha))
        w_alpha = np.where(w >= w_threshold, w, 0)
        I_alpha = np.count_nonzero(w_alpha)  # Size of set I(alhpa)
        w_sum = np.sum(w_alpha)  # sum of w in I(alpha)
        w_sum_square = np.sum(w_alpha ** 2)  # sum of squares of w in I(alpha)
        g_alpha = w_sum_square / 2 - w_sum / n + I_alpha * ((1 - alpha) ** 2 -\
                  (1 - delta) ** 2) / (2 * (n ** 2) * ( 1 - alpha) ** 2) +\
                  (n * (1 - delta) ** 2 - 2 * rho) / (2 * (n ** 2) * (1 - alpha) ** 2)

        if g_alpha < 0:
            alpha_up = alpha
        else:
            alpha_low = alpha

    alpha = (alpha_up + alpha_low) / 2

    return alpha


def FMD_p(x, p, i, step_alpha, delta, rho, alpha_tol, X_train,y_train,s_norm, RHS):
    """
    Full gradient descent on p
    """
    # for each constraint and JL sample one index I_t ~ p_t^i
    # If we are using chi-square as our f-divergence, we need to relax our uncertainty set.
    n, d = X_train.shape
    X_theta = X_train @ x
    if i == 0:
        grad_val = np.log(1 + np.exp(X_theta)) - np.multiply(1-y_train,X_theta) - RHS[i]
    elif i == 1:
        grad_val = np.multiply(s_norm,X_theta) - RHS[i]
    else:
        grad_val = np.multiply(-s_norm, X_theta) - RHS[i]

    p += step_alpha * grad_val
    # Projection to our chi-square uncertainty set
    # Here we dont use efficient projection scheme, which uses tree structure

    # Note that g'(alpha) is a decreasing function of alpha
    # Input w_sum and w_square_sum are sum of p[t] and square sum of p[t].

    alpha = find_alpha_full(p, n, delta, rho, alpha_tol)
    # Update p value according to alpha that we find above
    # For i != I_t, p_{t+1} = (1-alpha)w_i + alpha/n
    p *= (1-alpha)
    p += alpha/n

    # For i = I_t, we should first take a look at w_{t,I_t}
    p = np.where(p>delta/n,p, delta/n)

    return p

#FMD Solver

def DRO_FMD(x_0, p_0, X_train, y_train, delta, rho, alpha_tol,\
            opt_low, opt_up, obj_tol, RHS, ss_type,dual_gap_option, dual_gap_freq,dual_gap_freq2, print_option=1,\
            K_test_flag=0, min_flag=1, feas_opt = 0, warm_start = 0):
    """
        dual_gap_option: There are 4 different options for SP gap calcuation.
        dual_gap_freq: Frequency of computing the SP gap.
        min_flag: indicates whether our problem is minimization (=1) or maximization (=0).
        If dual_gap_option == 0: then don't calculate dual_gap and use absolute dual_gap_freq.
        If dual_gap_option == 1: then don't calculate dual_gap and use relative dual_gap_freq.
        # For this example only, if dual_gap_option == 0 or 1, then we also calculate the duality gap and see if it is less than equal to eps/2.
        We exclude duality gap calculation time from total_time. Also, turn off the hat_f calculation as we are not using for the algorithm of ICML.
        Turn on again for Journal Version later.
        If dual_gap_option == 2: then calculate dual_gap and use absolute dual_gap_freq.
        If dual_gap_option == 3: then calculate dual_gap and use relative dual_gap_freq.
    """

    n, d = X_train.shape
    s_norm = X_train[:,0] - np.mean(X_train[:,0])
    m = 3

    print('\n')
    print('************************************************************************************')
    print('*******************************Problem Description**********************************')
    print(' ')
    print('Number of constraints: %s, x dimension: %s ,Uncertainty Set Dimension: %s' % (m, d, n))
    print(' ')
    print('*************************************************************************************')
    print('*************************************************************************************')

    i_flag_count = 0

    if K_test_flag:
        print('We are doing sample size test here!')

    G = 0.25
    M = np.ones(3) * 0.25
    print('G:', G)
    print('M:', M)

    if ss_type == 'diminish':
        T, R_x, R_p, c_x, c_p = R_dim_FMD(d, n, G, M, rho, obj_tol)
        print("Max Iteration:", T)
        print('alg_type: FGD with i^*')
        ss_x_list = c_x * (np.sqrt(1 / (np.arange(T + 1) + 1)))
        ss_p_list = []
        for i in range(m):
            ss_p_list.append(c_p[i] * (np.sqrt(1 / (np.arange(T + 1) + 1))))
    print('Rx: %s' % R_x)
    print('Rp: %s' % R_p)

    bisection = []
    bisection_count = 0

    vartheta_list = []
    dual_gap_list = []  # each element is a list of dual gap at each bisection iteration
    iter_timer_list = []  # each element  elapsed time per bisection
    real_T_list = []  # each element is a list of terminated iteration by dual gap condition at each
    # bisection iteration
    early_term_count = 0
    dual_gap_time = 0 #Measures the time used for calculating duality gap
    solved_flag = 0 #Only used when dual_gap_cal = 0
    total_tic = time.time()

    sanity_check = 0
    sanity_freq = 1000


    if dual_gap_option == 2 or dual_gap_option == 0:
        pass
    elif dual_gap_option ==3 or dual_gap_option == 1:
        dual_gap_freq = int(T * dual_gap_freq)
        dual_gap_freq2 = int(T * dual_gap_freq2)

    #Set calculation option
    if dual_gap_option == 0 or dual_gap_option == 1:
        dual_gap_cal = 0
    else:
        dual_gap_cal = 1

    feas_flag = 0
    change_flag = 1
    # Implementing normal test
    if K_test_flag == 0:

        while opt_up - opt_low > obj_tol and not feas_flag:
            iter_tic = time.time()
            feas_flag = feas_opt

            break_flag = 0
            bisection_count += 1

            # Change our objective function
            obj_val = (opt_up + opt_low) / 2
            RHS[0] = (-1 + 2 * min_flag) * obj_val
            if print_option:
                print('---------------------------')
                print('%s-th bisection iteration' % bisection_count)
                print('alg_type: FGD with i^*')
                print('---------------------------')
                print("current step objective value:", obj_val)



            #Change Dual_Freq_Gap if warm_startup == 1 and bisection_count > 1
            # if warm_start and bisection_count > 1:
            #     dual_gap_freq = 10

            if warm_start and bisection_count > 1 and change_flag:
                dual_gap_freq = dual_gap_freq2
                change_flag = 0

            x = np.empty([dual_gap_freq, d])
            p = np.empty([dual_gap_freq, m, n])
            f_val = np.zeros([dual_gap_freq, m])
            f_val_ws = np.zeros(m)




            # Set initial point according to warm_start
            if warm_start:
                if bisection_count == 1:
                    pass
                else:
                    x_0 = deepcopy(x_ws / ss_sum_x)
                    p_0 = deepcopy(p_ws / ss_sum_p)

            x[0, :] = x_0
            p[0, :, :] = p_0


            # Variables that is needed to update periodically.
            iter_count = 0
            ss_sum_x = 0
            ss_sum_p = 0  # This does not have to be np.zeros([m])
            x_ws = np.zeros(d)
            p_ws = np.zeros([m, n])


            tic = time.time()

            dual_gap = []  # List that contains duality gap in this bisection




            for t in range(T):

                # Now our step-size is not uniform. Make sure to change x_bar and p_bar according to our step size

                if (t+2) % dual_gap_freq == 0 and print_option:
                    toc = time.time()
                    print('=========================================')
                    print('%s-st iteration start time:' % (t + 1), toc - tic)


                x[(t+1)%dual_gap_freq,:], f_val[t%dual_gap_freq,:] = FMD_x(x[t%dual_gap_freq,:], p[t%dual_gap_freq,:,:],\
                                                            ss_x_list[t], X_train, y_train, s_norm, RHS)

                for i in range(m):
                    p[(t+1)%dual_gap_freq,i,:] = FMD_p(x[t%dual_gap_freq,:], p[t%dual_gap_freq,i,:], i, ss_p_list[i][t],\
                                                       delta, rho, alpha_tol, X_train,y_train,s_norm, RHS)

                """

                Duality Gap Termination Condition(Implement when dual_flag = 1)

                """
                # Update bar_x and bar_p
                if dual_gap_cal == 0 and (t + 2) % dual_gap_freq == 0:
                    dual_gap_tic = time.time()
                    x_ws, ss_sum_x = bar_calculator_x(x_ws, x, dual_gap_freq, \
                                                      ss_x_list[
                                                      iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                                      ss_sum_x)
                    p_ws, ss_sum_p = bar_calculator_p(p_ws, p, dual_gap_freq, \
                                                      ss_p_list[0][
                                                      iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                                      ss_sum_p)
                    dual_gap_toc = time.time()
                    dual_gap_time += dual_gap_toc - dual_gap_tic
                    print('dual_gap_time:', dual_gap_time)

                # Calculate Dual gap
                if dual_gap_cal and (t+2) % dual_gap_freq == 0:
                    dual_gap_tic = time.time()
                    x_ws, ss_sum_x = bar_calculator_x(x_ws, x,dual_gap_freq,
                                        ss_x_list[iter_count * dual_gap_freq:(iter_count+1) * dual_gap_freq],ss_sum_x)
                    p_ws, ss_sum_p = bar_calculator_p(p_ws, p, dual_gap_freq,
                        ss_p_list[0][iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],ss_sum_p)
                    sup_val = sup_pi(x_ws / ss_sum_x, X_train, y_train, s_norm, rho, delta,
                                     alpha_tol, RHS)
                    inf_val = inf_pi(p_ws / ss_sum_p, X_train, y_train, s_norm, RHS)
                    diff = sup_val - inf_val
                    dual_gap.append(diff)
                    dual_gap_toc = time.time()

                if  dual_gap_cal and (t+2) % dual_gap_freq == 0 and print_option:

                    print("%s-st iteration duality gap:" % (t + 1), diff)
                    print("Dual Gap Calculation Time %s" % (dual_gap_toc - dual_gap_tic))
                    print("sup_val:", sup_val)
                    print("inf_val:", inf_val)
                    print('w_sum:',np.sum(p[(t+1)%dual_gap_freq,:,:],axis = 1)) #We need to turn this print option later.
                    p_temp = p[(t+1)%dual_gap_freq,:,:]
                    print('divergence:', np.sum((p_temp - 1/n)**2,axis=1) * n / 2)

                if (t + 1) % dual_gap_freq == 0:
                    iter_count += 1

                # If K_test_flag == 1, we don't use duality gap termination condition
                if dual_gap_cal and (t+2) % dual_gap_freq == 0 and diff <= obj_tol / 2:
                    real_t = t + 1
                    break_flag = 1
                    iter_toc = time.time()
                    dual_gap_list.append(dual_gap)
                    iter_timer_list.append(iter_toc - iter_tic)
                    real_T_list.append(t)
                    early_term_count += 1

                    if print_option:
                        print("=============================================")
                        if bisection_count == 11:
                            print("%s-th bisection iteration terminated early!!" % bisection_count)
                        elif bisection_count == 12:
                            print("%s-th bisection iteration terminated early!!" % bisection_count)
                        elif bisection_count == 13:
                            print("%s-th bisection iteration terminated early!!" % bisection_count)
                        elif bisection_count % 10 == 1:
                            print("%s-st bisection iteration terminated early!!" % bisection_count)
                        elif bisection_count % 10 == 2:
                            print("%s-nd bisection iteration terminated early!!" % bisection_count)
                        elif bisection_count % 10 == 3:
                            print("%s-rd bisection iteration terminated early!!" % bisection_count)
                        else:
                            print("%s-th bisection iteration terminated early!!" % bisection_count)
                        print("Terminated in %s iterations" % (t + 1))
                        print("Max iteration %s" % T)
                        print('x_bar:', x_ws/ss_sum_x)
                        # print('p_bar:', p_bar)
                        print('Duality Gap:', diff)
                        print("=============================================")

                    if pi_val(x_ws/ss_sum_x, p_ws/ss_sum_p, X_train, y_train, s_norm, RHS) > obj_tol / 2:
                        if min_flag:
                            opt_low = obj_val
                            bisection.append('low')
                        else:
                            opt_up = obj_val
                            bisection.append('up')
                    else:
                        if min_flag:
                            opt_up = obj_val
                            bisection.append('up')
                        else:
                            opt_low = obj_val
                            bisection.append('low')

                    break

                if dual_gap_cal and t == T - 1 and print_option:
                    real_t = T

                    x_ws, ss_sum_x = bar_calculator_x(x_ws, x, T + 1 - iter_count * dual_gap_freq, \
                                                         ss_x_list[iter_count * dual_gap_freq:], ss_sum_x)
                    p_ws, ss_sum_p = bar_calculator_p(p_ws, p, T + 1 - iter_count * dual_gap_freq, \
                                                         ss_p_list[0][iter_count * dual_gap_freq:], ss_sum_p)
                    sup_val = sup_pi(x_ws / ss_sum_x, X_train, y_train, s_norm, rho, delta,
                                     alpha_tol, RHS)
                    inf_val = inf_pi(p_ws / ss_sum_p, X_train, y_train, s_norm, RHS)
                    diff = sup_val - inf_val
                    dual_gap.append(diff)
                    print("%s-st iteration duality gap:" % (t + 1), diff)
                    print("sup_val:", sup_val)
                    print("inf_val:", inf_val)



            if dual_gap_cal == 0:
                real_t = T

                dual_gap_tic = time.time()
                # Calculate the duality gap at the last iteration if dual_gap_cal == 0
                x_ws, ss_sum_x = bar_calculator_x(x_ws, x, T + 1 - iter_count * dual_gap_freq, \
                                                  ss_x_list[iter_count * dual_gap_freq:], ss_sum_x)
                p_ws, ss_sum_p = bar_calculator_p(p_ws, p, T + 1 - iter_count * dual_gap_freq, \
                                                  ss_p_list[0][iter_count * dual_gap_freq:], ss_sum_p)
                sup_val = sup_pi(x_ws / ss_sum_x, X_train, y_train, s_norm, rho, delta, alpha_tol, RHS)
                inf_val = inf_pi(p_ws / ss_sum_p, X_train, y_train, s_norm, RHS)
                diff = sup_val - inf_val
                dual_gap.append(diff)
                print("%s-st iteration duality gap:" % T, diff)
                dual_gap_toc = time.time()
                dual_gap_time += dual_gap_toc - dual_gap_tic
                print('duality gap computation time:', dual_gap_time)
                if diff < obj_tol / 2:
                    solved_flag = 1


            dual_gap_list.append(dual_gap)
            real_T_list.append(t)

            if break_flag:
                continue
        total_toc = time.time()
        total_solved_time = total_toc - total_tic
        obj_val = (opt_up + opt_low) / 2

        print('Out of %s bisection iteration %s terminated early' % (bisection_count, early_term_count))
        print('Average iteration:', mean(real_T_list))
        print('==========================================')


    stat = Statistics(n, m, n, n, ss_type, obj_val, dual_gap_list,
                      iter_timer_list, \
                      total_solved_time, real_T_list, T, R_x, R_p, i_flag_count, 0,0,
                      solved_flag=solved_flag, dual_gap_time = dual_gap_time)

    # Update the last objective value

    # obj_val = (-1 + 2*min_flag) * obj_val

    return stat

def DRO_FMD_K_test_time(x_0, p_0,X_train, y_train, delta, rho, alpha_tol, opt_low, opt_up, obj_tol,RHS,
               ss_type, dual_gap_option, dual_gap_freq, time_cap, time_freq, print_option=1,  min_flag=1):
    """
     Function for comparing the convergence speed of SOFO and OFO based approach.
     We implement K_test_time with this function. At every time_freq, we calculate the dual gap and record
     its value.
     """
    n, d = X_train.shape
    s_norm = X_train[:,0] - np.mean(X_train[:,0])
    m = 3
    # Calculate coefficients

    print('\n')
    print('************************************************************************************')
    print('*******************************Problem Description**********************************')
    print(' ')
    print('Number of constraints: %s, x dimension: %s ,Uncertainty Set Dimension: %s' % (m, d, n))
    print(' ')
    print('*************************************************************************************')
    print('*************************************************************************************')

    i_flag_count = 0

    # We calculate G and M here
    G = 0.25
    M = np.ones(3) * 0.25
    print('G:', G)
    print('M:', M)

    if ss_type == 'diminish':
        T, R_x, R_p, c_x, c_p = R_dim_FMD(d, n, G, M, rho, obj_tol)
        print('alg_type: FGD with i^*')
        print("Max Iteration:", T)
        T_max = 1e7
        ss_x_list = c_x * (np.sqrt(1 / (np.arange(T_max + 1) + 1)))
        obj_tol = 1e-7
        ss_p_list = []
        for i in range(m):
            ss_p_list.append(c_p[i] * (np.sqrt(1 / (np.arange(T_max + 1) + 1))))

    print('Rx: %s' % R_x)
    print('Rp: %s' % R_p)

    # This String List would be used on inf_pi function
    constr_list = []
    for i in range(m):
        constr_list.append('obj_' + str(i))

    bisection = []
    bisection_count = 0

    vartheta_list = []
    dual_gap_list = []  # each element is a list of dual gap at each bisection iteration
    iter_timer_list = []  # each element  elapsed time per bisection
    real_T_list = []  # each element is a list of terminated iteration by dual gap condition at each
    # bisection iteration
    early_term_count = 0
    total_tic = time.time()

    # sample_freq = 10**5
    sanity_check = 0
    sanity_freq = 1000

    if dual_gap_option == 2 or dual_gap_option == 0:
        pass
    elif dual_gap_option == 3 or dual_gap_option == 1:
        dual_gap_freq = int(T * dual_gap_freq)

    # Set calculation option
    if dual_gap_option == 0 or dual_gap_option == 1:
        dual_gap_cal = 0
    else:
        dual_gap_cal = 1

    iter_tic = time.time()

    break_flag = 0
    bisection_count += 1

    # Change our objective function
    obj_val = (opt_up + opt_low) / 2
    RHS[0] = (-1 + 2 * min_flag) * obj_val
    if print_option:
        print('---------------------------')
        print('%s-th bisection iteration' % bisection_count)
        print('alg_type: FGD with i^*')
        print('---------------------------')
        print("current step objective value:", obj_val)


    x = np.empty([dual_gap_freq, d])
    p = np.empty([dual_gap_freq, m, n])
    f_val = np.zeros([dual_gap_freq, m])
    f_val_ws = np.zeros(m)

    # Variables that is needed to update periodically.
    x[0, :] = x_0
    p[0, :, :] = p_0
    iter_count = 0
    ss_sum_x = 0
    ss_sum_p = 0  # This does not have to be np.zeros([m])
    x_ws = np.zeros(d)
    p_ws = np.zeros([m, n])
    x_list = [x_0]
    p_list = [p_0]


    dual_gap = []  # List that contains duality gap in this bisection
    tic = time.time()
    toc = time.time()
    t = 0
    x_ws_list = []
    p_ws_list = []
    ss_sum_x_list = []
    ss_sum_p_list = []
    time_iter_count = 1

    while toc - tic < time_cap:

        toc = time.time()
        if print_option and toc - tic > time_iter_count * time_freq:
            print('=========================================')
            print('%s-st iteration start time:' % (t + 1), toc - tic)
            print('Time Iter Count: %s' % time_iter_count)
            x_list.append(x_ws/ss_sum_x)
            p_list.append(p_ws/ss_sum_p)
            time_iter_count += 1

        x[(t + 1) % dual_gap_freq, :], f_val[t % dual_gap_freq, :] = FMD_x(x[t % dual_gap_freq, :],
                                                                           p[t % dual_gap_freq, :, :], \
                                                                           ss_x_list[t], X_train, y_train, s_norm, RHS)

        for i in range(m):
            p[(t + 1) % dual_gap_freq, i, :] = FMD_p(x[t % dual_gap_freq, :], p[t % dual_gap_freq, i, :], i,
                                                     ss_p_list[i][t], \
                                                     delta, rho, alpha_tol, X_train, y_train, s_norm, RHS)
        """

        Duality Gap Termination Condition(Implement when dual_flag = 1)

        """
        # Calculate Dual gap



        if dual_gap_cal and (t + 2) % dual_gap_freq == 0:
            x_ws, ss_sum_x = bar_calculator_x(x_ws, x, dual_gap_freq, \
                                              ss_x_list[
                                              iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                              ss_sum_x)
            p_ws, ss_sum_p = bar_calculator_p(p_ws, p, dual_gap_freq, \
                                              ss_p_list[0][
                                              iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                              ss_sum_p)
        if (t + 1) % dual_gap_freq == 0:
            # f_val_ws += np.tensordot(f_val, ss_x_list[iter_count * dual_gap_freq: \
            #                                           (iter_count + 1) * dual_gap_freq], axes=(0, 0))
            iter_count += 1
        t+=1
    dual_gap_tic = time.time()
    for idx in range(len(x_list)):
        sup_val = sup_pi(x_list[idx], X_train, y_train, s_norm, rho, delta, alpha_tol, RHS)
        inf_val = inf_pi(p_list[idx], X_train, y_train, s_norm, RHS)
        diff = sup_val - inf_val
        if print_option:
            print('sup val:', sup_val)
            print('inf val:', inf_val)
            print('{}-th Dual Gap: {}'.format(idx, diff))
        dual_gap.append(diff)
    dual_gap_toc = time.time()
    dual_gap_time = dual_gap_toc - dual_gap_tic

    dual_gap_list.append(dual_gap)
    total_toc = time.time()
    total_solved_time = total_toc - total_tic

    FMD_stat = Statistics(n, m, n, n, ss_type, obj_val, dual_gap_list,
                          iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_flag_count, 0,0,solved_flag = 0, dual_gap_time=dual_gap_time)

    return FMD_stat



