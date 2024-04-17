import math
import numpy as np
import time 
import gurobipy as gp
from gurobipy import GRB
from statistics import mean
from FML_UBRegret import R_dim_FMD
from FML_utils import Statistics, bar_calculator_x, bar_calculator_p, sup_pi, inf_pi, pi_val
from copy import deepcopy

def FMD_x(x, p,step_alpha, X_train, y_train, s_norm, RHS):

    """
    Full Mirror Descent for x

    Args:
        x: current x
        p: current p
        step_alpha: step size
        X_train: training data
        y_train: labels
        s_norm: normalized sensitive feature
        RHS: right hand side of constraints
    """

    n, d = X_train.shape
    m = 3
    F_hat = []

    """
    i_hat calculation
    """
    p_sum = np.sum(p, axis = 1)
    X_theta = X_train @ x
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
    """
    Find optimal alpha value for the projection step
    """

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

def FMD_p(x, p, i, step_alpha, delta, rho, alpha_tol, X_train,y_train,s_norm, RHS):
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
    Args:
        x_0: initial x
        p_0: initial p
        X_train: training data
        y_train: labels
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
        dual_gap_freq2: dual gap frequency 2
        print_option: print option
        K_test_flag: K test flag
        min_flag: 1 if our problem is minimization problem
        feas_opt: feasibility option
        warm_start: warm start option
    
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

    dual_gap_list = []  # each element is a list of dual gap at each bisection iteration
    iter_timer_list = []  # each element  elapsed time per bisection
    real_T_list = []  # each element is a list of terminated iteration by dual gap condition at each
    # bisection iteration
    early_term_count = 0
    dual_gap_time = 0 #Measures the time used for calculating duality gap
    solved_flag = 0 #Only used when dual_gap_cal = 0
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
                    #print("x_bar:", x_bar)
                    # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
                    # print("p_new[0,0,0:]:", p_new[0,0,:])
                    # print("X Coeff:", temp_coeff)
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
                    # print("x_bar:", x_bar)
                    # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
                    # print("p_new[0,0,0:]:", p_new[0,0,:])
                    # print("X Coeff:", temp_coeff)
                    print("sup_val:", sup_val)
                    print("inf_val:", inf_val)
                    #print('w_sum:', np.sum(p[t + 1], axis=1))



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
                                                     delta, rho, alpha_tol, emp_dist_value_copy)

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
                      iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_flag_count, 0,0)

    return FMD_stat

def DRO_FMD_K_test_time(x_0, p_0,X_train, y_train, delta, rho, alpha_tol, opt_low, opt_up, obj_tol,RHS,
               ss_type, dual_gap_option, dual_gap_freq, time_cap, time_freq, print_option=1,  min_flag=1):
    

    """
    We calculate the dual gap and record its value for every dual_gap_freq.

    Args: 
        x_0: initial x
        p_0: initial p
        X_train: training data
        y_train: labels
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
        time_cap: time cap
        time_freq: time frequency
        print_option: print option
        min_flag: 1 if our problem is minimization problem
    
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



