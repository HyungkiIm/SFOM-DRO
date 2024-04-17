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
from MN_UBRegret import R_dim_FMD
from MN_utils import *
from copy import deepcopy

def FMD_x(c_vec, s_vec, r_vec, b_vec, beta,x, p,step_alpha, RHS, emp_dist_value, budget):
    """
    Full Mirror Descent for x

    Args:
        c_vec, s_vec, r_vec, b_vec, beta, budget: parameters
        x: current x
        p: current p
        step_alpha: step size
        emp_dist_value: training dataset
        RHS: right hand side of constraints
    """
    # Get the value of m,J,L,n
    d, n = emp_dist_value.shape
    d += 1
    m = 2
    g_t = np.zeros(d)
    F_hat = []

    """
    i_hat calculation
    """
    p_sum = np.sum(p, axis = 1)

    for i in range(m):
        if i == 0:
            temp = np.dot(p[i,:], get_L(c_vec, s_vec, r_vec, b_vec,x[1:],emp_dist_value))
            temp -= RHS[i] * p_sum[i]
            F_hat.append(temp)
        if i == 1:
            temp = np.dot(p[i,:],x[0] + 1 / beta * np.maximum(get_L(c_vec, s_vec, r_vec, b_vec,x[1:],emp_dist_value) - x[0],0))
            temp -= RHS[i] * p_sum[i]
            F_hat.append(temp)

    i_hat = F_hat.index(max(F_hat))
    """
    g_t calculation
    """
    if i_hat == 0:
        g_t[1:] = np.tensordot(p[i_hat,:], get_grad_L(c_vec, s_vec, r_vec, b_vec,x[1:],emp_dist_value), axes = (0,1))
    elif i_hat == 1:
        L_val = get_L(c_vec, s_vec, r_vec, b_vec, x[1:], emp_dist_value)
        L_grad = get_grad_L(c_vec, s_vec, r_vec, b_vec, x[1:], emp_dist_value)
        diff_vec = np.where(L_val > x[0], 1, 0)
        g_t[0] = p_sum[i_hat] - 1 / beta * p[i_hat,:].T @ diff_vec
        g_t[1:] = 1 / beta * np.multiply(diff_vec.reshape(1,-1), L_grad) @ p[i_hat, :]
        #g_t[1:] = 1 / beta * L_grad @ np.diag(diff_vec) @ p[i_hat,:]

    """
    x_{t+1} calculation
    """

    # Get next x (Projection Step with entropy prox function)
    x_update = np.zeros(d)
    x_update[0] = x[0] - step_alpha * g_t[0]
    theta = np.multiply(x[1:], np.exp(-step_alpha * g_t[1:]))
    sum_theta = np.sum(theta)
    if sum_theta >= budget:
        x_update[1:] = budget * theta / sum_theta
    else:
        x_update[1:] = theta

    return x_update, np.array(F_hat)


def find_alpha_full(w, n, delta, rho, tol):
    """
    Find optimal alpha value for the projection step
    """
    alpha_up = 1
    alpha_low = 0
    g_alpha = 1
    esp = 1e-7
    # First sort w vector
    while abs(g_alpha) > tol:

        alpha = (alpha_up + alpha_low) / 2

        if alpha <= esp or 1 - alpha <= esp:
            return alpha

        w_threshold = (delta - alpha) / (n * (1 - alpha))
        # print(w)
        w_alpha = np.where(w >= w_threshold, w, 0)
        # print(w_alpha)
        # print(w_threshold)
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

def FMD_p(c_vec, s_vec, r_vec, b_vec, beta,x, p, i, step_alpha, delta, rho, alpha_tol, emp_dist_value, RHS):
    # for each constraint and JL sample one index I_t ~ p_t^i
    # If we are using chi-square as our f-divergence, we need to relax our uncertainty set.

    d, n = emp_dist_value.shape
    d += 1

    if i == 0:
        grad_val = get_L(c_vec, s_vec, r_vec, b_vec,x[1:], emp_dist_value) - RHS[i]
    if i == 1:
        grad_val = x[0] + 1 / beta * np.maximum(get_L(c_vec, s_vec, r_vec, b_vec,x[1:], emp_dist_value) - x[0],0) - RHS[i]

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

def DRO_FMD(c_vec,s_vec, r_vec, b_vec, mean_vec, beta, budget,x_0, p_0, emp_dist_value, delta, rho, alpha_tol,\
            opt_low, opt_up, obj_tol, RHS, ss_type,dual_gap_option, dual_gap_freq,dual_gap_freq2, print_option=1,\
            K_test_flag=0, min_flag=1, feas_opt = 0, warm_start = 0):
    """
    Args:
        c_vec, s_vec, r_vec, b_vec, beta, budget: problem parameters
        x_0: initial x
        p_0: initial p
        emp_dist_value: training dataset
        delta: p \geq \delta/n
        rho: \rho value for the uncertainty set
        alpha_tol: tolerance for the alpha value
        opt_low: lower bound of the optimal value
        opt_up: upper bound of the optimal value
        obj_tol: tolerance of our objective value
        RHS: RHS of the constraints
        ss_type: step-size type
        dual_gap_option: dual gap option
                If dual_gap_option == 0: then don't calculate dual_gap and use absolute dual_gap_freq.
                If dual_gap_option == 1: then don't calculate dual_gap and use relative dual_gap_freq.
                    For Fairness ML example only, if dual_gap_option == 0 or 1, then we also calculate the duality gap and see if it is less than equal to eps/2 to guarantee optimality.
                    We exclude duality gap calculation time from total_time for fairness of comparison.
                If dual_gap_option == 2: then calculate dual_gap and use absolute dual_gap_freq.
                If dual_gap_option == 3: then calculate dual_gap and use relative dual_gap_freq.
        dual_gap_freq: dual gap frequency
        dual_gap_freq2: dual gap frequency for warm start
        print_option: print option
        K_test_flag: K test flag
        min_flag: 1 if our problem is minimization problem
        feas_opt: feasibility option
    """

    emp_dist_value_copy = deepcopy(emp_dist_value)
    d, n = emp_dist_value_copy.shape
    d += 1
    m = 2

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

    G = 1 / (2*beta) * max([np.absolute(c_vec - s_vec).max(), np.absolute(c_vec - b_vec - r_vec).max()])
    M = np.zeros(2)
    Loss_vec_bar = get_L(c_vec, s_vec, r_vec, b_vec, mean_vec, emp_dist_value_copy)
    tau_bar = np.quantile(Loss_vec_bar, 1 - beta)
    M[0] = np.absolute(Loss_vec_bar).max()
    M[1] = np.absolute(tau_bar + 1 / beta * (np.where(Loss_vec_bar > tau_bar, Loss_vec_bar - tau_bar, 0))).max()
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

    opt_model = gp.Model('PWL')

    opt_model.setParam('OutputFlag', 0)
    var_t = opt_model.addMVar(1, lb=-GRB.INFINITY, name='t')
    var_x = opt_model.addMVar(d - 1, name='x')
    var_tau = opt_model.addMVar(1, lb=-GRB.INFINITY, name='tau')
    var_y = opt_model.addMVar((d - 1, n), lb=-float('inf'), name='y')
    var_z = opt_model.addMVar(n, name='z')

    constr_list = []
    for i in range(m):
        constr_list.append('obj_' + str(i))

        # Create dummy constraints
    for i in range(m):
        opt_model.addConstr(var_t >= i, name=constr_list[i])

    opt_model.addConstr(var_x.sum() <= budget)
    opt_model.addConstrs(var_x >= var_y[:, j] for j in range(n))
    opt_model.addConstrs(var_y[:, j] <= emp_dist_value_copy[:, j] for j in range(n))
    opt_model.addConstrs(var_z[i:i + 1] >= (c_vec - s_vec).T @ var_x - (b_vec + r_vec - s_vec).T @ var_y[:, i] +
                         b_vec.T @ emp_dist_value_copy[:, i] - var_tau for i in range(n))
    obj = var_t[0]
    opt_model.setObjective(obj, GRB.MINIMIZE)
    opt_model.update()

    bisection = []
    bisection_count = 0

    vartheta_list = []
    dual_gap_list = []  # each element is a list of dual gap at each bisection iteration
    iter_timer_list = []  # each element  elapsed time per bisection
    real_T_list = []  # each element is a list of terminated iteration by dual gap condition at each
    # bisection iteration
    early_term_count = 0
    solved_flag = 0
    total_tic = time.time()


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
    total_solved_time = 0
    total_dual_gap_time = 0
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
            dual_gap = []  # List that contains duality gap in this bisection
            dual_gap_time = 0  # Measures the time used for calculating duality gap

            iter_tic = time.time()
            tic = time.time()
            print(f'total solved time with {bisection_count}-th bisection: {total_solved_time}')
            print(f'total solved time with {bisection_count}-th bisection without dual calculation time: {total_solved_time-total_dual_gap_time}')
            for t in range(T):

                # Now our step-size is not uniform. Make sure to change x_bar and p_bar according to our step size

                if (t+2) % dual_gap_freq == 0 and print_option:
                    toc = time.time()
                    print('=========================================')
                    print('%s-st iteration start time:' % (t + 1), toc - tic)


                x[(t+1)%dual_gap_freq,:], f_val[t%dual_gap_freq,:] = FMD_x(c_vec, s_vec, r_vec, b_vec, beta,x[t%dual_gap_freq,:], p[t%dual_gap_freq,:,:],\
                                                            ss_x_list[t], RHS,emp_dist_value_copy, budget)

                for i in range(m):
                    p[(t+1)%dual_gap_freq,i,:] = FMD_p(c_vec, s_vec, r_vec, b_vec, beta,x[t%dual_gap_freq,:], p[t%dual_gap_freq,i,:], i, ss_p_list[i][t],\
                                                       delta, rho, alpha_tol, emp_dist_value_copy, RHS)
                """

                Duality Gap Termination Condition(Implement when dual_flag = 1)

                """
                # Calculate Dual gap
                if dual_gap_cal and (t+2) % dual_gap_freq == 0:
                    dual_gap_tic = time.time()
                    x_ws, ss_sum_x = bar_calculator_x(x_ws, x,dual_gap_freq,
                                        ss_x_list[iter_count * dual_gap_freq:(iter_count+1) * dual_gap_freq],ss_sum_x)
                    p_ws, ss_sum_p = bar_calculator_p(p_ws, p, dual_gap_freq,
                        ss_p_list[0][iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],ss_sum_p)
                    sup_val = sup_pi(c_vec, s_vec, r_vec, b_vec, beta, x_ws / ss_sum_x, emp_dist_value_copy, rho, delta,
                                     alpha_tol, RHS)
                    inf_val = inf_pi(c_vec, s_vec, r_vec, b_vec, beta, p_ws / ss_sum_p, emp_dist_value_copy, RHS,
                                     opt_model, var_x, var_y, var_tau,
                                     var_z, var_t, constr_list)
                    diff = sup_val - inf_val
                    dual_gap.append(diff)
                    dual_gap_toc = time.time()
                    dual_gap_time += dual_gap_toc - dual_gap_tic
                    
                if  dual_gap_cal and (t+2) % dual_gap_freq == 0 and print_option:
                    
                    print("%s-st iteration duality gap:" % (t + 1), diff)
                    print("sup_val:", sup_val)
                    print("inf_val:", inf_val)
                    print('w_sum:',np.sum(p[(t+1)%dual_gap_freq,:,:],axis = 1)) #We need to turn this print option later.
                    print('dual_gap_time:', dual_gap_time)
                if (t + 1) % dual_gap_freq == 0:
                    f_val_ws += np.tensordot(f_val, ss_x_list[iter_count * dual_gap_freq: \
                                                                      (iter_count + 1) * dual_gap_freq], axes=(0, 0))
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

                    if pi_val(c_vec, s_vec, r_vec, b_vec, beta,x_ws/ss_sum_x, p_ws/ss_sum_p, emp_dist_value_copy, RHS) > obj_tol / 2:
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
                    dual_gap_tic = time.time()
                    x_ws, ss_sum_x = bar_calculator_x(x_ws, x, T + 1 - iter_count * dual_gap_freq, \
                                                         ss_x_list[iter_count * dual_gap_freq:], ss_sum_x)
                    p_ws, ss_sum_p = bar_calculator_p(p_ws, p, T + 1 - iter_count * dual_gap_freq, \
                                                         ss_p_list[0][iter_count * dual_gap_freq:], ss_sum_p)
                    sup_val = sup_pi(c_vec, s_vec, r_vec, b_vec, beta, x_ws / ss_sum_x, emp_dist_value_copy, rho, delta,
                                     alpha_tol, RHS)
                    inf_val = inf_pi(c_vec, s_vec, r_vec, b_vec, beta, p_ws / ss_sum_p, emp_dist_value_copy, RHS,
                                     opt_model, var_x, var_y, var_tau,
                                     var_z, var_t, constr_list)
                    diff = sup_val - inf_val
                    dual_gap.append(diff)
                    dual_gap_toc = time.time()
                    dual_gap_time += dual_gap_toc - dual_gap_tic
                    print("%s-st iteration duality gap:" % (t + 1), diff)
                    print("sup_val:", sup_val)
                    print("inf_val:", inf_val)


            if dual_gap_cal == 0:
                real_t = T

            dual_gap_list.append(dual_gap)
            real_T_list.append(t)
            iter_toc = time.time()
            total_solved_time += (iter_toc - iter_tic)
            total_dual_gap_time += dual_gap_time

            if break_flag:
                continue

            f_val_ws += np.tensordot(f_val[: T % dual_gap_freq], ss_x_list[iter_count * dual_gap_freq:T],
                                         axes=(0, 0))
            ss_sum_x = np.sum(ss_x_list[:T])
            f_val_ws /= ss_sum_x
            vartheta = f_val_ws.max()

            if vartheta > R_x:
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



        total_toc = time.time()
        total_solved_time = total_toc - total_tic
        obj_val = (opt_up + opt_low) / 2

        print('Out of %s bisection iteration %s terminated early' % (bisection_count, early_term_count))
        print('Average iteration:', mean(real_T_list))
        print(f'total solved time with {bisection_count} bisection: {total_solved_time}')
        print(f'total solved time with {bisection_count} bisection without dual calculation time: {total_solved_time-total_dual_gap_time}')
        print('==========================================')


    stat = Statistics(n, m, 0, 0, ss_type, obj_val, dual_gap_list,
                      iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_flag_count, 0, 0,
                      solved_flag=solved_flag, dual_gap_time=total_dual_gap_time)
    return stat
