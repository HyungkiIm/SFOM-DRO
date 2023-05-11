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
from UBRegret import R_dim_FMD, R_const_FMD
from utils import *

# Todo
"""
1. Go through Full_alpha function
2. Saving p value in tree structure?
3. For j in J for l in L improvement?
4. Memory error occuring. --> Do not calculate Duality Gap?
5. Updating f_val when dual_gap_freq == 0 is slow. How should we improve?
Especially tensordot calculation is really slow.. 
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
Here we split SMD and FMD solver into two different functions.(Splitting SMD Complete.)
Oct-06
When dual_gap_freq ==0, we don't save p_t anymore.
Changed for j in J, for l in L, using tensordot.


"""


def FMD_x(x, p, step_alpha, RHS, emp_dist_value):
    # Get the value of m,J,L,n
    m, J, L, n = emp_dist_value.shape
    g_t = np.zeros([J, L])  # subgradient estimator
    F_hat = []

    """
    i_hat calculation
    """

    for i in range(m):
        temp = 0
        temp = np.sum(np.multiply(np.tensordot(p[i, :], emp_dist_value[i, :, :, :], axes=(0, 2)), x))
        temp = temp - RHS[i]
        F_hat.append(temp)

    i_hat = F_hat.index(max(F_hat))

    """
    g_t calculation
    """

    g_t = np.tensordot(p[i_hat, :], emp_dist_value[i_hat, :, :, :], axes=(0, 2))

    """
    x_{t+1} calculation
    """

    # Get next x (Projection Step with entropy prox function)
    theta = np.multiply(x, np.exp(-step_alpha * g_t))
    x_update = np.zeros([J, L])
    for j in range(J):
        x_update[j, :] = theta[j, :] / np.sum(theta[j, :])

    return x_update


def find_alpha_full(w, n, delta, rho, tol):
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
        g_alpha = w_sum_square / 2 - w_sum / n + I_alpha * ((1 - alpha) ** 2 - \
                                                            (1 - delta) ** 2) / (2 * (n ** 2) * (1 - alpha) ** 2) + \
                  (n * (1 - delta) ** 2 - 2 * rho) / (2 * (n ** 2) * (1 - alpha) ** 2)

        if g_alpha < 0:
            alpha_up = alpha
        else:
            alpha_low = alpha

    alpha = (alpha_up + alpha_low) / 2

    return alpha


"""
Construct A_i for each i
We do not utilize tree structure here. 
prox function follows divergence of our problem
Here i is a index of constraint
"""


def FMD_p(x, p, i, step_alpha, delta, rho, alpha_tol, emp_dist_value):
    # for each constraint and JL sample one index I_t ~ p_t^i
    # If we are using chi-square as our f-divergence, we need to relax our uncertainty set.

    _, J, L, n = emp_dist_value.shape

    p += step_alpha * np.tensordot(x, emp_dist_value[i, :, :, :], axes=([0, 1], [0, 1]))
    # Projection to our chi-square uncertainty set
    # Here we dont use efficient projection scheme, which uses tree structure

    # Note that g'(alpha) is a decreasing function of alpha
    # Input w_sum and w_square_sum are sum of p[t] and square sum of p[t].

    alpha = find_alpha_full(p, n, delta, rho, alpha_tol)
    # Update p value according to alpha that we find above
    # For i != I_t, p_{t+1} = (1-alpha)w_i + alpha/n
    p *= (1 - alpha)
    p += alpha / n

    # For i = I_t, we should first take a look at w_{t,I_t}

    for index in range(n):
        if (1 - alpha) * p[index] + alpha / n < delta / n:
            p[index] = delta / n

    return p


# SMD Solver


def DRO_FMD(x_0, p_0, emp_dist_value, delta, rho, alpha_tol, opt_low, opt_up, obj_tol,
            RHS,
            ss_type, dual_gap_option, dual_gap_freq, T_cap, print_option=1, K_test_flag=0, min_flag=1):
    """
        If dual_gap_option == 0: then don't calculate dual_gap and use absolute dual_gap_freq.
        If dual_gap_option == 1: then don't calculate dual_gap and use relative dual_gap_freq.
        If dual_gap_option == 2: then calculate dual_gap and use absolute dual_gap_freq.
        If dual_gap_option == 3: then calculate dual_gap and use relative dual_gap_freq.
    """

    # Change data according to min_flag.
    emp_dist_value_copy = emp_dist_value.copy()
    emp_dist_value_copy[0, :, :, :] = (-1 + 2 * min_flag) * emp_dist_value_copy[0, :, :, :]

    m, J, L, n = emp_dist_value.shape  # Parameters

    # Calculate coefficients

    C_G = 1 + math.sqrt(rho / n)

    print('\n')
    print('************************************************************************************')
    print('*******************************Problem Description**********************************')
    print(' ')
    print('Number of constraints: %s, x dimension: %s ,Uncertainty Set Dimension: %s' % (m, J * L, n))
    print(' ')
    print('*************************************************************************************')
    print('*************************************************************************************')

    i_flag_count = 0

    if K_test_flag:
        print('We are doing sample size test here!')

    # Change C_K according to K value

    # Calculate stoc_factor if K_grad < n
    # Currently we just use stoc_factor = 1

    # Omega = np.log(2*m/nu)
    #
    # if K_grad < n:
    #     stoc_factor = 1 + Omega/2 + 4*math.sqrt(Omega)

    stoc_factor = 1  # We need to adjust this part later Sep-20

    # G is bound of norm of gradient(\nabla) F_k^i(x) for all k \in [n] and i \in [m] and x \in X
    G = np.absolute(emp_dist_value).max()  # Also, need to change this funciton add C2
    # M is an list of bound of |F_k^i(x)| for each i \in [m]
    absolute_max = np.absolute(emp_dist_value).max(axis=2)
    sum_over_J = np.sum(absolute_max, axis=1)
    M = sum_over_J.max(axis=1)

    # Calculate T and our stepsize
    if ss_type == 'constant':
        T, R_x, R_p, ss_x, ss_p = R_const_FMD(J, L, n, G, M, rho, obj_tol)
        if K_test_flag:
            print("Max Iteration:", T_cap)
            print('alg_type: FGD with i^*')
            T = T_cap
            obj_tol = 1e-7
            ss_x_list = ss_x * np.ones(T + 1)
            ss_p_list = []
            for i in range(m):
                ss_p_list.append(ss_p[i] * np.ones(T + 1))
        else:
            print("Max Iteration:", T)
            print('alg_type: FGD with i^*')
            ss_x_list = ss_x * np.ones(T + 1)
            ss_p_list = []
            for i in range(m):
                ss_p_list.append(ss_p[i] * np.ones(T + 1))



    elif ss_type == 'diminish':
        T, R_x, R_p, c_x, c_p = R_dim_FMD(J, L, n, G, M, rho, obj_tol)
        if K_test_flag:
            print("Max Iteration:", T_cap)
            print('alg_type: FGD with i^*')
            T = T_cap
            ss_x_list = c_x * (np.sqrt(1 / (np.arange(T + 1) + 1)))
            obj_tol = 1e-7
            ss_p_list = []
            for i in range(m):
                ss_p_list.append(c_p[i] * (np.sqrt(1 / (np.arange(T + 1) + 1))))

        else:
            print("Max Iteration:", T)
            print('alg_type: FGD with i^*')
            ss_x_list = c_x * (np.sqrt(1 / (np.arange(T + 1) + 1)))
            ss_p_list = []
            for i in range(m):
                ss_p_list.append(c_p[i] * (np.sqrt(1 / (np.arange(T + 1) + 1))))
    print('Rx: %s' % R_x)
    print('Rp: %s' % R_p)

    # This String List would be used on inf_pi function
    constr_list = []
    for i in range(m):
        constr_list.append('obj_' + str(i))

    bisection = []
    bisection_count = 0

    vartheta_list = []

    # x_coeff,_ = get_coeff(emp_dist_value_copy,rho,delta, alpha_tol)
    # print(x_coeff)
    # Define a PWL to calculate inf_pi
    list_J = list(range(J))
    list_L = list(range(L))

    PWL_model = gp.Model('PWL')

    PWL_model.setParam('OutputFlag', 0)
    var_t = PWL_model.addVar(lb=-GRB.INFINITY, name='t')
    var_x = PWL_model.addVars(list_J, list_L, ub=1, name='x')
    PWL_model.addConstrs(gp.quicksum(var_x[j, l] for l in list_L) == 1 for j in list_J)

    # Create dummy constraints
    for i in range(m):
        PWL_model.addConstr(var_t >= i, name=constr_list[i])

    obj = var_t

    PWL_model.setObjective(obj, GRB.MINIMIZE)
    PWL_model.optimize()

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

    # Implementing normal test
    if K_test_flag == 0:

        while opt_up - opt_low > obj_tol:
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

            x = []
            p = []

            if dual_gap_freq == 0:
                p_old = p_0
                f_val = np.zeros(m)
            else:
                p_tmp = np.zeros([m, n])

            x.append(x_0)
            p.append(p_0)
            # Get x and p for T iterations
            tic = time.time()

            dual_gap = []  # List that contains duality gap in this bisection
            if dual_gap_freq == 0:

                for i in range(m):
                    f_val[i] += np.sum(np.multiply(np.tensordot(p_old[i, :], emp_dist_value_copy[i, :, :, :], \
                                                                axes=(0, 2)), x[0])) * ss_x_list[0]

            for t in range(T):

                # Now our step-size is not uniform. Make sure to change x_bar and p_bar according to our step size

                if dual_gap_freq != 0 and t % dual_gap_freq == 0 and print_option:
                    toc = time.time()
                    print('=========================================')
                    print('%s-st iteration start time:' % (t + 1), toc - tic)

                if dual_gap_freq != 0:

                    x_new = FMD_x(x[t], p[t], ss_x_list[t], RHS, emp_dist_value_copy)
                    x.append(x_new)

                    for i in range(m):
                        p_tmp[i, :] = FMD_p(x[t], p[t][i, :], i, ss_p_list[i][t], delta, rho, alpha_tol, \
                                            emp_dist_value_copy)

                        # if abs(np.sum(p_new) - w_sum_temp) > 1e-3:
                        #     print('sum of p:', np.sum(p_new))
                        #     print('w_sum:', w_sum_temp)
                        #     raise TypeError("There is significant differnence")

                    p.append(p_tmp)

                else:

                    x_new = FMD_x(x[t], p_old, ss_x_list[t], RHS, emp_dist_value_copy)
                    x.append(x_new)

                    for i in range(m):
                        p_new_i = FMD_p(x[t], p_old[i, :], i,
                                        ss_p_list[i][t], delta, rho, alpha_tol, emp_dist_value_copy)
                        p_old[i, :] = p_new_i

                        f_val[i] += np.sum(np.multiply(np.tensordot(p_old[i, :], emp_dist_value_copy[i, :, :, :], \
                                                                    axes=(0, 2)), x[t])) * ss_x_list[t]

                # Sanity Check

                if dual_gap_freq == 0 and sanity_check:
                    p.append(p_old)
                    if t % sanity_freq == 0:
                        x_bar = bar_calculator(x, t + 2, ss_x_list)
                        p_bar = bar_calculator(p, t + 2, ss_p_list[0])
                        sup_val = sup_pi(x_bar, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
                        inf_val = inf_pi(p_bar, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
                        diff = sup_val - inf_val
                        dual_gap.append(diff)
                        print("%s-st iteration duality gap:" % (t + 1), diff)
                        # print("x_bar:", x_bar)
                        # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
                        # print("p_new[0,0,0:]:", p_new[0,0,:])
                        # print("X Coeff:", temp_coeff)
                        print("sup_val:", sup_val)
                        print("inf_val:", inf_val)
                        print('w_sum:', np.sum(p[t + 1], axis=1))  # We need to turn this print option later.

                """

                Duality Gap Termination Condition(Implement when dual_flag = 1)

                """
                # Calculate Dual gap
                if dual_gap_freq != 0 and t % dual_gap_freq == 0:
                    x_bar = bar_calculator(x, t + 2, ss_x_list)
                    p_bar = bar_calculator(p, t + 2, ss_p_list[0])
                    sup_val = sup_pi(x_bar, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
                    inf_val = inf_pi(p_bar, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
                    diff = sup_val - inf_val
                    dual_gap.append(diff)

                if dual_gap_freq != 0 and t % dual_gap_freq == 0 and print_option:
                    print("%s-st iteration duality gap:" % (t + 1), diff)
                    # print("x_bar:", x_bar)
                    # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
                    # print("p_new[0,0,0:]:", p_new[0,0,:])
                    # print("X Coeff:", temp_coeff)
                    print("sup_val:", sup_val)
                    print("inf_val:", inf_val)
                    print('w_sum:', np.sum(p[t + 1], axis=1))  # We need to turn this print option later.

                test_freq = 1000
                if dual_gap_freq == 0 and print_option and t % test_freq == 0:
                    toc = time.time()
                    print('=========================================')
                    print('%s-st iteration start time:' % (t + 1), toc - tic)
                    print('w_sum at %s iteration: %s' % (t, np.sum(p_old, axis=1)))
                    # We need to turn this print option later.

                if dual_gap_freq != 0 and t == T - 1:
                    x_bar = bar_calculator(x, t + 2, ss_x_list)
                    p_bar = bar_calculator(p, t + 2, ss_p_list[0])
                    sup_val = sup_pi(x_bar, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
                    inf_val = inf_pi(p_bar, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
                    diff = sup_val - inf_val
                    dual_gap.append(diff)
                    print("%s-st iteration duality gap:" % (t + 1), diff)
                    # print("x_bar:", x_bar)
                    # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
                    # print("p_new[0,0,0:]:", p_new[0,0,:])
                    # print("X Coeff:", temp_coeff)
                    print("sup_val:", sup_val)
                    print("inf_val:", inf_val)
                    print('w_sum:', np.sum(p[t + 1], axis=1))

                # If K_test_flag == 1, we don't use duality gap termination condition
                if dual_gap_freq != 0 and t % dual_gap_freq == 0 and diff <= obj_tol / 2:
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
                        print('x_bar:', x_bar)
                        # print('p_bar:', p_bar)
                        print('Duality Gap:', diff)
                        print("=============================================")
                    if pi_val(x_bar, p_bar, emp_dist_value_copy, RHS) > obj_tol / 2:
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

            dual_gap_list.append(dual_gap)

            if break_flag:
                continue

            real_T_list.append(t)

            if dual_gap_freq != 0:
                f_val = np.zeros(m)

                # Todo
                # np.asarray is slow, any alternative? Then we should use np.array at the start of this code
                # Fix this memory error today.. as well as temp_tic and temp_toc.
                p = np.asarray(p)
                x = np.asarray(x)
                temp_tic = time.time()
                for i in range(m):
                    f_val[i] = np.average(np.sum(np.multiply(np.tensordot(p[:, i, :], emp_dist_value_copy[i, :, :, :], \
                                                                          axes=(1, 2)), x), axis=(1, 2)),
                                          weights=ss_x_list)
                temp_toc = time.time()
                f_val = f_val - RHS
                vartheta = f_val.max()


            else:

                f_val /= np.sum(ss_x_list)
                f_val = f_val - RHS
                vartheta = f_val.max()

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
        print('==========================================')

    # elif K_test_flag == 1:
    #     bisection_count += 1
    #
    #     # Change our objective function
    #     obj_val = (opt_up + opt_low) / 2
    #     RHS[0] = (-1 + 2 * min_flag) * obj_val
    #     if print_option:
    #         print('---------------------------')
    #         print('%s-th bisection iteration' % bisection_count)
    #         print('alg_type:')
    #         print('---------------------------')
    #         print("current step objective value:", obj_val)
    #
    #     x = []
    #     p = []
    #     x.append(x_0)
    #     p.append(p_0)
    #     # Get x and p for T iterations
    #     tic = time.time()
    #
    #     w_sum = np.zeros(m)
    #     w_sum[:] = np.nan
    #     w_square_sum = np.zeros(m)
    #     w_square_sum[:] = np.nan
    #     cum_dist = np.zeros([m, n])
    #     cum_dist[:] = np.nan
    #     for i in range(m):
    #         w_sum[i] = np.sum(p_0[i, :])
    #         w_square_sum[i] = np.sum(p_0[i, :] ** 2)
    #         cum_dist[i, :] = np.cumsum(p_0[i, :])
    #
    #     dual_gap = []  # List that contains duality gap in this bisection
    #
    #     for t in range(T):
    #
    #         # Now our step-size is not uniform. Make sure to change x_bar and p_bar according to our step size
    #
    #         if t % dual_gap_freq == 0 and print_option:
    #             toc = time.time()
    #             print('=========================================')
    #             print('%s-st iteration start time:' % (t + 1), toc - tic)
    #         x_new, i_flag = x_alg(K, K_grad, x[t], p[t], prox, divg, ss_x_list[t], RHS, emp_dist_value_copy, \
    #                               random_samples_list[t, :, :], w_sum, cum_dist, C_K, obj_tol)
    #         x.append(x_new)
    #         i_flag_count += i_flag
    #         tmp = np.zeros([m, n])
    #         for i in range(m):
    #             p_new, w_sum_temp, w_square_sum_temp, cum_dist_temp = p_alg(x[t], p[t], i, divg, ss_p_list[i][t],
    #                                                                         delta,
    #                                                                         rho, alpha_tol, \
    #                                                                         emp_dist_value_copy, w_sum[i],
    #                                                                         w_square_sum[i], cum_dist[i, :])
    #             w_sum[i] = w_sum_temp
    #             w_square_sum[i] = w_square_sum_temp
    #             cum_dist[i, :] = cum_dist_temp
    #             # if abs(np.sum(p_new) - w_sum_temp) > 1e-3:
    #             #     print('sum of p:', np.sum(p_new))
    #             #     print('w_sum:', w_sum_temp)
    #             #     raise TypeError("There is significant differnence")
    #             tmp[i, :] = p_new
    #
    #         p.append(tmp)
    #
    #         # Calculate Dual gap
    #         if t % dual_gap_freq == 0:
    #             x_bar = bar_calculator(x, t + 2, ss_x_list)
    #             p_bar = bar_calculator(p, t + 2, ss_p_list[0])
    #             sup_val = sup_pi(x_bar, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
    #             inf_val = inf_pi(p_bar, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
    #             diff = sup_val - inf_val
    #             dual_gap.append(diff)
    #
    #     dual_gap_list.append(dual_gap)
    #
    #     total_toc = time.time()
    #     total_solved_time = total_toc - total_tic
    #     obj_val = (opt_up + opt_low) / 2

    stat = Statistics(n, m, n, n, ss_type, x_bar, p_bar, obj_val, x, p, dual_gap_list,
                      iter_timer_list, \
                      total_solved_time, real_T_list, T, R_x, R_p, i_flag_count)

    # Update the last objective value

    # obj_val = (-1 + 2*min_flag) * obj_val

    return stat