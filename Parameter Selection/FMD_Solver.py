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

#Todo
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



def FMD_x(x, p,step_alpha, RHS, emp_dist_value):
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
        If dual_gap_option == 0: then don't calculate dual_gap and use absolute dual_gap_freq.
        If dual_gap_option == 1: then don't calculate dual_gap and use relative dual_gap_freq.
        If dual_gap_option == 2: then calculate dual_gap and use absolute dual_gap_freq.
        If dual_gap_option == 3: then calculate dual_gap and use relative dual_gap_freq.
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

    #sample_freq = 10**5
    sanity_check = 0
    sanity_freq = 1000


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

            for t in range(T):

                # Now our step-size is not uniform. Make sure to change x_bar and p_bar according to our step size

                if (t+2) % dual_gap_freq == 0 and print_option:
                    toc = time.time()
                    print('=========================================')
                    print('%s-st iteration start time:' % (t + 1), toc - tic)

                x[(t+1)%dual_gap_freq,:,:], f_val[t%dual_gap_freq,:] = FMD_x(x[t%dual_gap_freq,:,:], p[t%dual_gap_freq,:,:],\
                                                            ss_x_list[t], RHS,emp_dist_value_copy)

                for i in range(m):
                    p[(t+1)%dual_gap_freq,i,:] = FMD_p(x[t%dual_gap_freq,:,:], p[t%dual_gap_freq,i,:], i, ss_p_list[i][t],\
                                                       delta, rho, alpha_tol, emp_dist_value_copy, RHS)

                # #Sanity Check
                #
                # if dual_gap_freq == 0  and sanity_check:
                #     p.append(p_old)
                #     if t % sanity_freq == 0:
                #         x_bar = bar_calculator(x, t + 2, ss_x_list)
                #         p_bar = bar_calculator(p, t + 2, ss_p_list[0])
                #         sup_val = sup_pi(x_bar, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
                #         inf_val = inf_pi(p_bar, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
                #         diff = sup_val - inf_val
                #         dual_gap.append(diff)
                #         print("%s-st iteration duality gap:" % (t + 1), diff)
                #         # print("x_bar:", x_bar)
                #         # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
                #         # print("p_new[0,0,0:]:", p_new[0,0,:])
                #         # print("X Coeff:", temp_coeff)
                #         print("sup_val:", sup_val)
                #         print("inf_val:", inf_val)
                #         print('w_sum:', np.sum(p[t + 1], axis=1))  # We need to turn this print option later.




                """

                Duality Gap Termination Condition(Implement when dual_flag = 1)

                """
                # Calculate Dual gap
                if dual_gap_cal and (t+2) % dual_gap_freq == 0:
                    x_ws, ss_sum_x = bar_calculator_temp(x_ws, x,dual_gap_freq,
                                        ss_x_list[iter_count * dual_gap_freq:(iter_count+1) * dual_gap_freq],ss_sum_x)
                    p_ws, ss_sum_p = bar_calculator_temp(p_ws, p, dual_gap_freq,
                        ss_p_list[0][iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],ss_sum_p)
                    sup_val = sup_pi(x_ws / ss_sum_x, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
                    inf_val = inf_pi(p_ws / ss_sum_p, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
                    diff = sup_val - inf_val
                    dual_gap.append(diff)

                if  dual_gap_cal and (t+2) % dual_gap_freq == 0 and print_option:
                    print("%s-st iteration duality gap:" % (t + 1), diff)
                    #print("x_bar:", x_bar)
                    # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
                    # print("p_new[0,0,0:]:", p_new[0,0,:])
                    # print("X Coeff:", temp_coeff)
                    print("sup_val:", sup_val)
                    print("inf_val:", inf_val)
                    print('w_sum:',np.sum(p[(t+1)%dual_gap_freq,:,:],axis = 1)) #We need to turn this print option later.

                if (t + 1) % dual_gap_freq == 0:
                    f_val_ws += np.tensordot(f_val, ss_x_list[iter_count * dual_gap_freq: \
                                                                      (iter_count + 1) * dual_gap_freq], axes=(0, 0))
                    iter_count += 1

                # test_freq = 1000
                # if  dual_gap_freq == 0 and print_option and t% test_freq==0:
                #     toc = time.time()
                #     print('=========================================')
                #     print('%s-st iteration start time:' % (t + 1), toc - tic)
                #     print('w_sum at %s iteration: %s' %(t,np.sum(p_old,axis = 1)))
                #     #We need to turn this print option later.




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
                    x_ws, ss_sum_x = bar_calculator_temp(x_ws, x, T + 1 - iter_count * dual_gap_freq, \
                                                         ss_x_list[iter_count * dual_gap_freq:], ss_sum_x)
                    p_ws, ss_sum_p = bar_calculator_temp(p_ws, p, T + 1 - iter_count * dual_gap_freq, \
                                                         ss_p_list[0][iter_count * dual_gap_freq:], ss_sum_p)
                    sup_val = sup_pi(x_ws / ss_sum_x, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
                    inf_val = inf_pi(p_ws / ss_sum_p, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
                    diff = sup_val - inf_val
                    dual_gap.append(diff)
                    print("%s-st iteration duality gap:" % (t + 1), diff)
                    # print("x_bar:", x_bar)
                    # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
                    # print("p_new[0,0,0:]:", p_new[0,0,:])
                    # print("X Coeff:", temp_coeff)
                    print("sup_val:", sup_val)
                    print("inf_val:", inf_val)
                    #print('w_sum:', np.sum(p[t + 1], axis=1))



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
                      total_solved_time, real_T_list, T, R_x, R_p, i_flag_count)

    # Update the last objective value

    # obj_val = (-1 + 2*min_flag) * obj_val

    return stat

#This Function is used for K_test_iter
def DRO_FMD_K_test_iter(x_0, p_0, emp_dist_value, delta, rho, alpha_tol, opt_low, opt_up, obj_tol,RHS,
               ss_type, dual_gap_option, dual_gap_freq, T_cap, print_option=1,  min_flag=1):

    emp_dist_value_copy = emp_dist_value.copy()
    emp_dist_value_copy[0, :, :, :] = (-1 + 2 * min_flag) * emp_dist_value_copy[0, :, :, :]
    RHS[0] = (-1 + 2 * min_flag) * RHS[0]

    m, J, L, n = emp_dist_value.shape  # Parameters
    K = n
    K_grad = n
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


    # G is bound of norm of gradient(\nabla) F_k^i(x) for all k \in [n] and i \in [m] and x \in X
    G = np.absolute(emp_dist_value).max()  # Also, need to change this funciton add C2
    # M is an list of bound of |F_k^i(x)| for each i \in [m]
    absolute_max = np.absolute(emp_dist_value).max(axis=2)
    sum_over_J = np.sum(absolute_max, axis=1)
    M = sum_over_J.max(axis=1)

    # Calculate T and our stepsize
    if ss_type == 'constant':
        T, R_x, R_p, ss_x, ss_p = R_const_FMD(J, L, n, G, M, rho, obj_tol)
        print("Max Iteration:", T_cap)
        print('alg_type: FGD with i^*')
        T = T_cap
        obj_tol = 1e-7
        ss_x_list = ss_x * np.ones(T + 1)
        ss_p_list = []
        for i in range(m):
            ss_p_list.append(ss_p[i] * np.ones(T + 1))




    elif ss_type == 'diminish':
        T, R_x, R_p, c_x, c_p = R_dim_FMD(J, L, n, G, M, rho, obj_tol)
        print("Max Iteration:", T_cap)
        print('alg_type: FGD with i^*')
        T = T_cap
        ss_x_list = c_x * (np.sqrt(1 / (np.arange(T + 1) + 1)))
        obj_tol = 1e-7
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

    iter_tic = time.time()

    break_flag = 0
    bisection_count += 1

    # Change our objective function
    obj_val = (opt_up + opt_low) / 2

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


    sup_val = sup_pi(x_0, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
    inf_val = inf_pi(p_0, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
    diff = sup_val - inf_val
    dual_gap.append(diff)

    for t in range(T):

        # Now our step-size is not uniform. Make sure to change x_bar and p_bar according to our step size

        if (t + 2) % dual_gap_freq == 0 and print_option:
            toc = time.time()
            print('=========================================')
            print('%s-st iteration start time:' % (t + 1), toc - tic)

        x[(t + 1) % dual_gap_freq, :, :], f_val[t % dual_gap_freq, :] = FMD_x(x[t % dual_gap_freq, :, :],
                                                                              p[t % dual_gap_freq, :, :], \
                                                                              ss_x_list[t], RHS, emp_dist_value_copy)

        for i in range(m):
            p[(t + 1) % dual_gap_freq, i, :] = FMD_p(x[t % dual_gap_freq, :, :], p[t % dual_gap_freq, i, :], i,
                                                     ss_p_list[i][t], \
                                                     delta, rho, alpha_tol, emp_dist_value_copy,RHS)

        """

        Duality Gap Termination Condition(Implement when dual_flag = 1)

        """
        # Calculate Dual gap
        if dual_gap_cal and (t + 2) % dual_gap_freq == 0:
            x_ws, ss_sum_x = bar_calculator_temp(x_ws, x, dual_gap_freq,
                                                 ss_x_list[iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                                 ss_sum_x)
            p_ws, ss_sum_p = bar_calculator_temp(p_ws, p, dual_gap_freq,
                                                 ss_p_list[0][
                                                 iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq], ss_sum_p)
            sup_val = sup_pi(x_ws / ss_sum_x, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
            inf_val = inf_pi(p_ws / ss_sum_p, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
            diff = sup_val - inf_val
            dual_gap.append(diff)

        if dual_gap_cal and (t + 2) % dual_gap_freq == 0 and print_option:
            print("%s-st iteration duality gap:" % (t + 1), diff)
            # print("x_bar:", x_bar)
            # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
            # print("p_new[0,0,0:]:", p_new[0,0,:])
            # print("X Coeff:", temp_coeff)
            print("sup_val:", sup_val)
            print("inf_val:", inf_val)
            print('w_sum:',
                  np.sum(p[(t + 1) % dual_gap_freq, :, :], axis=1))  # We need to turn this print option later.

        if (t + 1) % dual_gap_freq == 0:
            f_val_ws += np.tensordot(f_val, ss_x_list[iter_count * dual_gap_freq: \
                                                      (iter_count + 1) * dual_gap_freq], axes=(0, 0))
            iter_count += 1

    dual_gap_list.append(dual_gap)
    total_toc = time.time()
    total_solved_time = total_toc - total_tic

    FMD_stat = Statistics(n, m, K, K_grad, ss_type, obj_val, dual_gap_list,
                      iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_flag_count)

    return FMD_stat

def DRO_FMD_K_test_time(x_0, p_0, emp_dist_value, delta, rho, alpha_tol, opt_low, opt_up, obj_tol,RHS,
               ss_type, dual_gap_option, dual_gap_freq, time_cap, time_freq, print_option=1,  min_flag=1):

    emp_dist_value_copy = emp_dist_value.copy()
    emp_dist_value_copy[0, :, :, :] = (-1 + 2 * min_flag) * emp_dist_value_copy[0, :, :, :]
    RHS[0] = (-1 + 2 * min_flag) * RHS[0]

    m, J, L, n = emp_dist_value.shape  # Parameters
    K = n
    K_grad = n
    # Calculate coefficients

    print('\n')
    print('************************************************************************************')
    print('*******************************Problem Description**********************************')
    print(' ')
    print('Number of constraints: %s, x dimension: %s ,Uncertainty Set Dimension: %s' % (m, J * L, n))
    print(' ')
    print('*************************************************************************************')
    print('*************************************************************************************')

    i_flag_count = 0


    # G is bound of norm of gradient(\nabla) F_k^i(x) for all k \in [n] and i \in [m] and x \in X
    G = np.absolute(emp_dist_value).max()  # Also, need to change this funciton add C2
    # M is an list of bound of |F_k^i(x)| for each i \in [m]
    absolute_max = np.absolute(emp_dist_value).max(axis=2)
    sum_over_J = np.sum(absolute_max, axis=1)
    M = sum_over_J.max(axis=1)

    # Calculate T and our stepsize
    if ss_type == 'constant':
        T, R_x, R_p, ss_x, ss_p = R_const_FMD(J, L, n, G, M, rho, obj_tol)
        print('alg_type: FGD with i^*')
        T_max = 1e7
        print("Max Iteration:", T)
        obj_tol = 1e-7
        ss_x_list = ss_x * np.ones(T_max + 1)
        ss_p_list = []
        for i in range(m):
            ss_p_list.append(ss_p[i] * np.ones(T_max + 1))




    elif ss_type == 'diminish':
        T, R_x, R_p, c_x, c_p = R_dim_FMD(J, L, n, G, M, rho, obj_tol)
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

    iter_tic = time.time()

    break_flag = 0
    bisection_count += 1

    # Change our objective function
    obj_val = (opt_up + opt_low) / 2

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

    dual_gap = []  # List that contains duality gap in this bisection


    sup_val = sup_pi(x_0, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
    inf_val = inf_pi(p_0, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
    diff = sup_val - inf_val
    dual_gap.append(diff)
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
            x_ws_list.append(x_ws.copy())
            p_ws_list.append(p_ws.copy())
            ss_sum_x_list.append(ss_sum_x)
            ss_sum_p_list.append(ss_sum_p)
            time_iter_count += 1


        x[(t + 1) % dual_gap_freq, :, :], f_val[t % dual_gap_freq, :] = FMD_x(x[t % dual_gap_freq, :, :],
                                                                              p[t % dual_gap_freq, :, :], \
                                                                              ss_x_list[t], RHS, emp_dist_value_copy)
        for i in range(m):
            p[(t + 1) % dual_gap_freq, i, :] = FMD_p(x[t % dual_gap_freq, :, :], p[t % dual_gap_freq, i, :], i,
                                                     ss_p_list[i][t], \
                                                     delta, rho, alpha_tol, emp_dist_value_copy, RHS)

        """

        Duality Gap Termination Condition(Implement when dual_flag = 1)

        """
        # Calculate Dual gap



        if dual_gap_cal and (t + 2) % dual_gap_freq == 0:
            x_ws, ss_sum_x = bar_calculator_temp(x_ws, x, dual_gap_freq,
                                                 ss_x_list[iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                                 ss_sum_x)
            p_ws, ss_sum_p = bar_calculator_temp(p_ws, p, dual_gap_freq,
                                                 ss_p_list[0][
                                                 iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq], ss_sum_p)

        if (t + 1) % dual_gap_freq == 0:
            f_val_ws += np.tensordot(f_val, ss_x_list[iter_count * dual_gap_freq: \
                                                      (iter_count + 1) * dual_gap_freq], axes=(0, 0))
            iter_count += 1
        t+=1

    for idx in range(len(x_ws_list)):
        sup_val = sup_pi(x_ws_list[idx] / ss_sum_x_list[idx], emp_dist_value_copy, rho, delta, alpha_tol, RHS)
        inf_val = inf_pi(p_ws_list[idx] / ss_sum_p_list[idx], emp_dist_value_copy, RHS, PWL_model, var_t, var_x,
                         constr_list)
        diff = sup_val - inf_val
        dual_gap.append(diff)

    dual_gap_list.append(dual_gap)
    total_toc = time.time()
    total_solved_time = total_toc - total_tic

    FMD_stat = Statistics(n, m, K, K_grad, ss_type, obj_val, dual_gap_list,
                          iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_flag_count)

    return FMD_stat



