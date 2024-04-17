import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import gurobipy as gp
from gurobipy import GRB
from statistics import mean
from tqdm import tqdm
from PS_UBRegret import R_dim_FMD, R_const_FMD
from PS_utils import Statistics, bar_calculator, sup_pi, inf_pi, pi_val

def FMD_x(x, p,step_alpha, RHS, emp_dist_value):
    """
    Full Mirror Descent for x

    Args:
        x: current x
        p: current p
        step_alpha: step size
        emp_dist_value: training dataset
        RHS: right hand side of constraints
    """

    # Get the value of m,J,L,n
    m, J, L, n = emp_dist_value.shape
    g_t = np.zeros([J, L])  # subgradient estimator
    F_hat = []

    """
    i_hat calculation
    """
    p_sum = np.sum(p, axis = 1)

    for i in range(m):
        temp = 0
        temp = np.sum(np.multiply(np.tensordot(p[i, :], emp_dist_value[i, :, :, :], axes=(0, 2)), x))
        temp = temp - RHS[i] * p_sum[i]
        F_hat.append(temp)





    i_hat = F_hat.index(max(F_hat))

    """
    g_t calculation
    """

    g_t = np.tensordot(p[i_hat,:],emp_dist_value[i_hat,:,:,:], axes = (0,2))

    """
    x_{t+1} calculation
    """

    # Get next x (Projection Step with entropy prox function)
    theta = np.multiply(x, np.exp(-step_alpha * g_t))
    x_update = np.zeros([J, L])
    for j in range(J):
        x_update[j, :] = theta[j, :] / np.sum(theta[j, :])

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


"""
Construct A_i for each i
We do not utilize tree structure here. 
prox function follows divergence of our problem
Here i is a index of constraint
"""

def FMD_p(x, p, i, step_alpha, delta, rho, alpha_tol, emp_dist_value, RHS):
    # for each constraint and JL sample one index I_t ~ p_t^i
    # If we are using chi-square as our f-divergence, we need to relax our uncertainty set.

    _, J, L, n = emp_dist_value.shape

    p += step_alpha * (np.tensordot(x,emp_dist_value[i,:,:,:], axes = ([0,1],[0,1])) - RHS[i])
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
    # for index in range(n):
    #     if (1 - alpha) * p[index] + alpha / n < delta / n:
    #         p[index] = delta / n

    return p

#SMD Solver


def DRO_FMD(x_0, p_0, emp_dist_value, delta, rho, alpha_tol, opt_low, opt_up, obj_tol,
               RHS,
               ss_type,dual_gap_option, dual_gap_freq, T_cap, print_option=1, K_test_flag=0, min_flag=1, feas_opt = 0):
    """
    Args:
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
        print_option: print option
        K_test_flag: K test flag
        min_flag: 1 if our problem is minimization problem
        feas_opt: feasibility option
    """

    #Change data according to min_flag.
    emp_dist_value_copy = emp_dist_value.copy()
    emp_dist_value_copy[0, :, :, :] = (-1 + 2 * min_flag) * emp_dist_value_copy[0, :, :, :]



    m, J, L, n = emp_dist_value.shape  # Parameters

    # Calculate coefficients

    C_G = 1 + math.sqrt(rho/n)

    print('\n')
    print('************************************************************************************')
    print('*******************************Problem Description**********************************')
    print(' ')
    print('Number of constraints: %s, x dimension: %s ,Uncertainty Set Dimension: %s' % (m, J * L, n))
    print(' ')
    print('*************************************************************************************')
    print('*************************************************************************************')


    i_flag_count = 0

    stoc_factor = 1 

    # G is bound of norm of gradient(\nabla) F_k^i(x) for all k \in [n] and i \in [m] and x \in X
    G = np.absolute(emp_dist_value).max()  # Also, need to change this funciton add C2
    # M is an list of bound of |F_k^i(x)| for each i \in [m]
    absolute_max = np.absolute(emp_dist_value).max(axis = 2)
    sum_over_J = np.sum(absolute_max, axis = 1)
    M = sum_over_J.max(axis = 1)




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
            print('alg_type: FGD with i^*' )
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

    if dual_gap_option == 2 or dual_gap_option == 0:
        pass
    elif dual_gap_option ==3 or dual_gap_option == 1:
        dual_gap_freq = int(T * dual_gap_freq)

    #Set calculation option
    if dual_gap_option == 0 or dual_gap_option == 1:
        dual_gap_cal = 0
    else:
        dual_gap_cal = 1

    feas_flag = 0
    # Implementing normal test

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

        x = np.empty([dual_gap_freq, J, L])
        x[0, :, :] = x_0
        p = np.empty([dual_gap_freq, m, n])
        p[0, :, :] = p_0

        f_val = np.zeros([dual_gap_freq, m])
        f_val_ws = np.zeros(m)

        # Variables that is needed to update periodically.
        iter_count = 0
        ss_sum_x = 0
        ss_sum_p = 0  # This does not have to be np.zeros([m])
        x_ws = np.zeros([J, L])
        p_ws = np.zeros([m, n])


        tic = time.time()

        dual_gap = []  # List that contains duality gap in this bisection
        total_dual_gap_time = 0
        for t in range(T):

            # Now our step-size is not uniform. Make sure to change x_bar and p_bar according to our step size

            if (t+2) % dual_gap_freq == 0 and print_option:
                toc = time.time()
                print('=========================================')
                print('%s-st iteration start time:' % (t + 1), toc - tic)
                print('%s-st iteration start time excluding dual_gap_time:' % (t + 1), toc - tic - total_dual_gap_time)

            x[(t+1)%dual_gap_freq,:,:], f_val[t%dual_gap_freq,:] = FMD_x(x[t%dual_gap_freq,:,:], p[t%dual_gap_freq,:,:],\
                                                        ss_x_list[t], RHS,emp_dist_value_copy)

            for i in range(m):
                p[(t+1)%dual_gap_freq,i,:] = FMD_p(x[t%dual_gap_freq,:,:], p[t%dual_gap_freq,i,:], i, ss_p_list[i][t],\
                                                    delta, rho, alpha_tol, emp_dist_value_copy, RHS)

            """

            Duality Gap Termination Condition(Implement when dual_flag = 1)

            """
            # Calculate Dual gap
            if dual_gap_cal and (t+2) % dual_gap_freq == 0:
                dual_gap_tic = time.time()
                x_ws, ss_sum_x = bar_calculator(x_ws, x,dual_gap_freq,
                                    ss_x_list[iter_count * dual_gap_freq:(iter_count+1) * dual_gap_freq],ss_sum_x)
                p_ws, ss_sum_p = bar_calculator(p_ws, p, dual_gap_freq,
                    ss_p_list[0][iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],ss_sum_p)
                sup_val = sup_pi(x_ws / ss_sum_x, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
                inf_val = inf_pi(p_ws / ss_sum_p, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
                diff = sup_val - inf_val
                dual_gap.append(diff)
                dual_gap_toc = time.time()
                total_dual_gap_time += (dual_gap_toc-dual_gap_tic)

            if  dual_gap_cal and (t+2) % dual_gap_freq == 0 and print_option:
                print("%s-st iteration duality gap:" % (t + 1), diff)
                print("Dual Gap Calculation Time %s" % (dual_gap_toc - dual_gap_tic))
                print("Total Dual Gap Calculation Time %s" % total_dual_gap_time)
                print("sup_val:", sup_val)
                print("inf_val:", inf_val)
                print('w_sum:',np.sum(p[(t+1)%dual_gap_freq,:,:],axis = 1)) #We need to turn this print option later.

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
                if pi_val(x_ws/ss_sum_x, p_ws/ss_sum_p, emp_dist_value_copy, RHS) > obj_tol / 2:
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
                x_ws, ss_sum_x = bar_calculator(x_ws, x, T + 1 - iter_count * dual_gap_freq, \
                                                        ss_x_list[iter_count * dual_gap_freq:], ss_sum_x)
                p_ws, ss_sum_p = bar_calculator(p_ws, p, T + 1 - iter_count * dual_gap_freq, \
                                                        ss_p_list[0][iter_count * dual_gap_freq:], ss_sum_p)
                sup_val = sup_pi(x_ws / ss_sum_x, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
                inf_val = inf_pi(p_ws / ss_sum_p, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
                diff = sup_val - inf_val
                dual_gap.append(diff)
                print("%s-st iteration duality gap:" % (t + 1), diff)
                print("sup_val:", sup_val)
                print("inf_val:", inf_val)
                dual_gap_toc = time.time()
                total_dual_gap_time += (dual_gap_toc - dual_gap_tic)



        if dual_gap_cal == 0:
            real_t = T

        dual_gap_list.append(dual_gap)
        real_T_list.append(t)

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
    print('==========================================')


    stat = Statistics(n, m, n, n, ss_type, obj_val, dual_gap_list,
                      iter_timer_list, \
                      total_solved_time, real_T_list, T, R_x, R_p, i_flag_count,
                      dual_gap_time = total_dual_gap_time)

    return stat


