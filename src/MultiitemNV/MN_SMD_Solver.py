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
from MN_UBRegret import R_dim_SMD
from MN_utils import Statistics, bar_calculator_x, bar_calculator_p, dual_gap_calculator, pi_val, get_L, get_grad_L
from copy import deepcopy
from cython_RBTree import RedBlackTree

def SMD_x_Tree(c_vec, s_vec, r_vec, b_vec, beta, K, K_grad, x, step_alpha,p_tree_list, coin_K_arr, RHS, emp_dist_value,\
          budget, w_sum):
    """
    Stochastic Mirror Descent for x using RBTree

    Args:
        c_vec: c vector
        s_vec: s vector
        r_vec: r vector
        b_vec: b vector
        beta: beta value
        budget: budget value
        K: per-iteration sample size
        K_grad: batch size of Stochastic Mirror Descent
        x: current x
        step_alpha: step size
        p_tree_list: list of trees
        coin_K_arr: random coin array for random sampling
        emp_dist_value: training dataset
        RHS: RHS of the constraints
        w_sum: sum of p[t]
        
    """



    # Get the value of m,J,L,n
    d, n = emp_dist_value.shape
    d += 1
    m = 2
    F_hat = []

    """
    i_hat calculation
    """
    # Get K_samples
    K_samples = []
    for i in range(m):
        K_samples.append(p_tree_list[i].random_sample(coin_K_arr[i, :, :]))

    """
    Compute hat(F)^{i,K} for each constraint i
    hat(F)^{i,K} = 1/K* sum_{j=1}^K F^i(x_t,z^{i,I_j^i})
    """
    for i in range(m):
        if i == 0:
            temp = w_sum[i] * np.sum(get_L(c_vec, s_vec, r_vec, b_vec,x[1:],emp_dist_value[:,K_samples[i]]))
            temp = temp / K - RHS[i] * w_sum[i]
            F_hat.append(temp)
        if i == 1:
            Loss_vec = get_L(c_vec, s_vec, r_vec, b_vec,x[1:],emp_dist_value[:,K_samples[i]])
            temp = w_sum[i] * np.sum(x[0] + 1 / beta * np.maximum(Loss_vec - x[0], 0))
            temp = temp / K - RHS[i] * w_sum[i]
            F_hat.append(temp)
    # get the index of max value in list F_hat
    i_hat = F_hat.index(max(F_hat))
    """
    g_t calculation
    """


    K_grad_samples = K_samples[i_hat][:K_grad]
    g_t = np.zeros(d)  # subgradient estimator

    if i_hat == 0:
        for k_idx in range(K_grad):
            g_t[1:] += w_sum[i_hat] * get_grad_L(c_vec, s_vec, r_vec, b_vec, x[1:], \
                    emp_dist_value[:,int(K_grad_samples[k_idx])]) / K_grad
    elif i_hat == 1:
        for k_idx in range(K_grad):
            L_grad = get_grad_L(c_vec, s_vec, r_vec, b_vec, x[1:], \
                    emp_dist_value[:,int(K_grad_samples[k_idx])])
            L_val = get_L(c_vec, s_vec, r_vec, b_vec, x[1:], \
                emp_dist_value[:,int(K_grad_samples[k_idx])])

            if L_val > x[0]:
                g_t[0] += w_sum[i_hat] * (1- 1 / beta) / K_grad
                g_t[1:] += w_sum[i_hat] * 1 / beta * L_grad
            else:
                g_t[0] += w_sum[i_hat]

    """
    x_{t+1} calculation
    """
    # Get next x (Projection Step with entropy prox function)
    x_update = np.zeros(d)
    x_update[0] = x[0] - step_alpha * g_t[0]
    theta = np.multiply(x[1:], np.exp(-step_alpha*g_t[1:]))
    sum_theta = np.sum(theta)
    if sum_theta >= budget:
        x_update[1:] = budget * theta / sum_theta
    else:
        x_update[1:] = theta

    return x_update, np.array(F_hat)

def find_alpha(p_val, w_temp, n, delta, rho, w_sum, w_square_sum):
    """
    Find optimal alpha value for the projection step
    """
    # Our input w_sum and w_square_sum is not updated. So we need to update before we proceed.

    alpha_up = 1
    alpha_low = 0
    g_alpha = 1

    # values when w_temp>= w_threshold, I(alpha) = [n]
    w_sum2 = w_sum - p_val + w_temp
    w_square_sum2 = w_square_sum - p_val ** 2 + w_temp ** 2
    I_alpha2 = n

    # values when w_temp< w_threshold, I(alpha) = [n]\[I_t]
    w_sum1 = w_sum - p_val
    w_square_sum1 = w_square_sum - p_val ** 2
    I_alpha1 = n - 1

    if w_temp >= delta / n:
        if w_square_sum2 / 2 - w_sum2 / n + 1 / (2 * n) < rho / n ** 2:
            alpha = 0
            return alpha
        else:
            alpha = 1 - math.sqrt(rho / (n ** 2 * (w_square_sum2 / 2 - w_sum2 / n + 1 / (2 * n))))
            return alpha

    alpha_thre = (delta / n - w_temp) / (1 / n - w_temp)

    while True:

        alpha = (alpha_up + alpha_low) / 2

        if alpha >= alpha_thre:  # I(alpha) = [n]
            w_sum = w_sum2
            w_square_sum = w_square_sum2
            I_alpha = I_alpha2
        else:  # I(alpha) = [n]\I_t
            w_sum = w_sum1
            w_square_sum = w_square_sum1
            I_alpha = I_alpha1

        # Calculate g'(alpha)
        g_alpha = w_square_sum / 2 - w_sum / n + I_alpha * ((1 - alpha) ** 2 - (1 - delta) ** 2) / (2 * (n ** 2) * \
                                                                                                    (
                                                                                                                1 - alpha) ** 2) + (
                              n * (1 - delta) ** 2 - 2 * rho) / (2 * (n ** 2) * (1 - alpha) ** 2)

        # Update the interval according to the g'(alpha) value
        if g_alpha < 0:
            alpha_up = alpha
        else:
            alpha_low = alpha

        # termination condition
        if alpha_low > alpha_thre:  # I(alpha) = [n]
            if w_square_sum2 / 2 - w_sum2 / n + 1 / (2 * n) < rho / n ** 2:
                alpha = 0
                return alpha
            else:
                alpha = 1 - math.sqrt(rho / (n ** 2 * (w_square_sum2 / 2 - w_sum2 / n + 1 / (2 * n))))
                # raise TypeError('Returning case 1')
                return alpha
        elif alpha_up <= alpha_thre:
            if w_square_sum1 / 2 - w_sum1 / n + (n - 1) / (2 * n ** 2) <= rho / n ** 2 - (1 - delta) ** 2 / (
                    2 * n ** 2):
                alpha = 0
                return alpha
            else:
                alpha = 1 - math.sqrt((rho / n ** 2 - (1 - delta) ** 2 / (2 * n ** 2)) / (
                            w_square_sum1 / 2 - w_sum1 / n + (n - 1) / (2 * n ** 2)))
                # raise TypeError('Returning case 2')
                return alpha

def SMD_p_Tree(c_vec, s_vec, r_vec, b_vec, beta, \
          x, p_tree, p_arr, coin_list, i, step_alpha, delta, rho, RHS, emp_dist_value, w_sum, w_square_sum):
    """
    update p using RBTree
    """

    d, n = emp_dist_value.shape
    d += 1


    #Sample an index
    I_t = p_tree.random_sample(coin_list.reshape(1, -1))[0]
    multi = p_tree.multi
    addi = p_tree.addi

    old_pval = multi * p_arr[I_t] + addi

    if i == 0: #Quadratic function gradient Value
        grad_val = (get_L(c_vec, s_vec, r_vec, b_vec,x[1:],emp_dist_value[:,I_t])-RHS[i]) * w_sum / old_pval
    elif i == 1:
        grad_val = (x[0] + 1 / beta * np.maximum(get_L(c_vec, s_vec, r_vec, b_vec,x[1:],emp_dist_value[:,I_t]) - x[0],0)\
                    - RHS[i]) * w_sum / old_pval
    # update p_t to w_t
    w_temp = old_pval + step_alpha * grad_val
    # Projection to our chi-square uncertainty set
    # Note that g'(alpha) is a decreasing function of alpha
    # Input w_sum and w_square_sum are sum of p[t] and square sum of p[t].
    alpha = find_alpha(old_pval, w_temp, n, delta, rho, w_sum, w_square_sum)
    adjusted_w = w_temp * (1 - alpha) + alpha / n
    # Update multi and addi.
    p_tree.multi = multi * (1 - alpha)
    p_tree.addi = addi * (1 - alpha) + alpha / n
    if adjusted_w < delta / n:
        p_tree.delete_node(p_arr[I_t], I_t)
        p_tree.insert_node((delta / n - p_tree.addi) / p_tree.multi, I_t)
        w_square_sum = (1 - alpha) ** 2 * (w_square_sum - old_pval ** 2) + 2 * (1 - alpha) * alpha * \
                       (w_sum - old_pval) / n + (n - 1) * alpha ** 2 / n ** 2 + delta ** 2 / n ** 2
        w_sum = (1 - alpha) * (w_sum - old_pval) + (n - 1) * alpha / n + delta / n
        p_arr[I_t] = (delta / n - p_tree.addi) / p_tree.multi
    else:  # p_new[I_t] > delta/n
        p_tree.delete_node(p_arr[I_t], I_t)
        p_tree.insert_node((adjusted_w - p_tree.addi) / p_tree.multi, I_t)
        p_arr[I_t] = (adjusted_w - p_tree.addi) / p_tree.multi
        w_sum += step_alpha * grad_val
        w_square_sum = w_square_sum - old_pval ** 2 + adjusted_w ** 2  # Recheck this part later.
        w_square_sum = (1 - alpha) ** 2 * w_square_sum + 2 * alpha * (1 - alpha) * w_sum / n + alpha ** 2 / n
        w_sum = (1 - alpha) * w_sum + alpha

    return p_arr, w_sum, w_square_sum


def DRO_SMD(c_vec,s_vec, r_vec, b_vec, mean_vec, beta, budget, x_0, p_0, emp_dist_value, K, K_grad, delta, rho, alpha_tol, opt_low, opt_up, obj_tol,
        RHS,ss_type, C_K, dual_gap_option, dual_gap_freq,dual_gap_freq2, print_option=1, K_test_flag=0, min_flag=1,feas_opt = 0,
            warm_start = 0):
    """
    Args:
        c_vec, s_vec, r_vec, b_vec: c,s,r,b vectors
        mean_vec: mean vector
        beta: beta value
        budget: budget value
        x_0: initial point of x
        p_0: initial point of p
        emp_dist_value: training dataset
        K: per-iteration sample size
        K_grad: batch size of Stochastic Mirror Descent
        delta: p \geq \delta/n
        rho: rho value for the uncertainty set
        alpha_tol: tolerance for the alpha value
        opt_low: lower bound of the optimal value
        opt_up: upper bound of the optimal value
        obj_tol: tolerance of our objective value
        RHS: RHS of the constraints
        ss_type: step-size type
        C_K: proportion of epsilon that we need to pay by using i_hat
        dual_gap_option: option for calculating dual gap
                If dual_gap_option == 0: then don't calculate dual_gap and use absolute dual_gap_freq.
                If dual_gap_option == 1: then don't calculate dual_gap and use relative dual_gap_freq.
                    For Fairness ML example only, if dual_gap_option == 0 or 1, then we also calculate the duality gap and see if it is less than equal to eps/2 to guarantee optimality.
                    We exclude duality gap calculation time from total_time for fairness of comparison.
                If dual_gap_option == 2: then calculate dual_gap and use absolute dual_gap_freq.
                If dual_gap_option == 3: then calculate dual_gap and use relative dual_gap_freq.
        dual_gap_freq: frequency of calculating dual gap
        dual_gap_freq2: frequency of calculating dual gap after the first dual_gap_freq
        print_option: print option
        K_test_flag: flag for K_test
        min_flag: flag for minimization
        feas_opt: feasibility option
    """

    emp_dist_value_copy = deepcopy(emp_dist_value)
    d, n = emp_dist_value_copy.shape
    d += 1
    m = 2


    # Calculate coefficients

    C_G = 1 + math.sqrt(rho / n)

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

    stoc_factor = 1  # 1 is good enough.
    # G is bound of norm of gradient(\nabla) F_k^i(x) for all k \in [n] and i \in [m] and x \in X
    # In Risk-Averse example, we set g to square of 90-th quantile of xi

    # We calculate G and M here
    G = 1 / (2*beta) * max([np.absolute(c_vec - s_vec).max(), np.absolute(c_vec - b_vec - r_vec).max()])
    M = np.zeros(2)
    Loss_vec_bar = get_L(c_vec, s_vec, r_vec, b_vec, mean_vec, emp_dist_value_copy)
    tau_bar = np.quantile(Loss_vec_bar, 1 - beta)
    M[0] = np.absolute(Loss_vec_bar).max()
    M[1] = np.absolute(tau_bar + 1 / beta * (np.where(Loss_vec_bar > tau_bar, Loss_vec_bar - tau_bar, 0))).max()
    print('G:', G)
    print('M:', M)

    if ss_type == 'diminish':
        T, R_x, R_p, c_x, c_p = R_dim_SMD(d, n, G, M, delta, rho, obj_tol, C_G, C_K,
                                          stoc_factor)
        print("Max Iteration:", T)
        print('alg_type:SGD with i_hat')
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
    solved_flag = 0  # Only used when dual_gap_cal = 0

    # Define Optimization Model
    opt_model = gp.Model('PWL')
    opt_model.setParam('OutputFlag', 0)
    opt_model.ModelSense = GRB.MINIMIZE
    var_t = opt_model.addMVar(1,lb=-GRB.INFINITY, name='t')
    var_x = opt_model.addMVar(d - 1, name='x')
    var_tau = opt_model.addMVar(1,lb=-GRB.INFINITY, name='tau')
    var_y = opt_model.addMVar((d-1,n), lb=-float('inf'), name='y')
    var_z = opt_model.addMVar(n, name='z')



    constr_list = []
    for i in range(m):
        constr_list.append('obj_' + str(i))
    # Create dummy constraints
    for i in range(m):
        opt_model.addConstr(var_t >= i, name=constr_list[i])

    

    opt_model.addConstr(var_x.sum() <= budget)
    opt_model.addConstrs(var_x >= var_y[:,j] for j in range(n))
    opt_model.addConstrs(var_y[:, j] <= emp_dist_value_copy[:, j] for j in range(n))
    opt_model.addConstrs(var_z[i:i+1] >= (c_vec-s_vec).T @ var_x - (b_vec + r_vec - s_vec).T @ var_y[:,i] +
                         b_vec.T @ emp_dist_value_copy[:, i] - var_tau for i in range(n))
    obj = var_t[0]
    opt_model.setObjective(obj, GRB.MINIMIZE)
    opt_model.update()
    sample_freq = 10 ** 4  # We can move this to parameter.

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


    #Check Whether hat_vartheta is approximating well.
    vartheta_flag = 0
    feas_flag = 0
    change_flag = 1
    total_solved_time = 0
    total_dual_gap_time = 0
    # Implementing normal test
    if K_test_flag == 0:
        while opt_up - opt_low > obj_tol and not feas_flag:
            feas_flag = feas_opt
            break_flag = 0
            bisection_count += 1
            # Change our objective function
            obj_val = (opt_up + opt_low) / 2
            RHS[0] = (-1 + 2 * min_flag) * obj_val
            if print_option:
                print('---------------------------')
                print('%s-th bisection iteration' % bisection_count)
                print('alg_type:SGD with i_hat')
                print('---------------------------')
                print("current step objective value:", obj_val)

            #If dual_gap_freq == 0, then we do not save previous output
            #Todo
            #We are saving x_t only. Exclude this step when it is needed.

            if warm_start and bisection_count > 1 and change_flag:
                dual_gap_freq = dual_gap_freq2
                change_flag = 0

            x = np.empty([dual_gap_freq,d])
            p = np.empty([dual_gap_freq, m, n])
            hat_f_val = np.zeros([dual_gap_freq,m])
            hat_f_val_ws = np.zeros(m)
            # multi keep tracks of multi values for each tree
            multi = np.empty([dual_gap_freq, m])
            multi[0, :] = 1
            # addi keep tracks of addi values for each tree
            addi = np.empty([dual_gap_freq, m])
            addi[0, :] = 0

            # Initialize the Tree
            p_tree_list = []
            # bst.construct intial weight equal to p_0
            if warm_start and bisection_count!=1:
                for i in range(m):
                    bst=RedBlackTree()
                    bst.construct_tree_by_custom_weights_py(p_ws[i,:]/ss_sum_p)
                    p_tree_list.append(bst)
            else:
                for i in range(m):
                    bst = RedBlackTree()
                    bst.construct_tree_py(n)
                    p_tree_list.append(bst)

            #Set initial point according to warm_start
            if warm_start:
                if bisection_count == 1:
                    pass
                else:
                    x_0 = deepcopy(x_ws/ss_sum_x)
                    p_0 = deepcopy(p_ws/ss_sum_p)

            x[0, :] = x_0
            p[0, :,:] = p_0
            # Variables that is needed to update periodically.
            iter_count = 0
            ss_sum_x = 0
            ss_sum_p = 0  # This does not have to be np.zeros([m])
            x_ws = np.zeros(d)
            p_ws = np.zeros([m, n])

            if vartheta_flag:
                real_f_val = np.zeros([dual_gap_freq, m])
                real_f_val_ws = np.zeros(m)

            # Get x and p for T iterations
            w_sum = np.zeros(m)
            w_sum[:] = np.nan
            w_square_sum = np.zeros(m)
            w_square_sum[:] = np.nan

            for i in range(m):
                w_sum[i] = np.sum(p_0[i, :])
                w_square_sum[i] = np.sum(p_0[i, :] ** 2)

            dual_gap = []  # List that contains duality gap in this bisection
            dual_gap_time = 0  # Measures the time used for calculating duality gap
            coin_K_arr = np.random.uniform(size=[sample_freq, m, K, math.ceil(2 * np.log2(n + 1))])
            coin_p_list = np.random.uniform(size=[sample_freq, math.ceil(2 * np.log2(n + 1))])
            iter_tic = time.time()
            tic = time.time()
            print(f'total solved time with {bisection_count}-th bisection: {total_solved_time}')
            print(f'total solved time with {bisection_count}-th bisection without dual calculation time: {total_solved_time-total_dual_gap_time}')
            for t in range(T):
                if print_option and (t+2)%dual_gap_freq ==0:
                    toc = time.time()
                    print('=========================================')
                    print('%s-st iteration start time:' % (t + 1), toc - tic)
                #We do not need information of p_t here. Only cumdist and w_sum is needed.
                # Update x_t
                x[(t+1)%dual_gap_freq,:],hat_f_val[t%dual_gap_freq,:] = SMD_x_Tree(c_vec, s_vec, r_vec, b_vec, beta, \
                    K, K_grad, x[t%dual_gap_freq,:],\
                    ss_x_list[t],p_tree_list,coin_K_arr[t % sample_freq,:, :, :],RHS, \
                        emp_dist_value_copy, budget, w_sum)
                # Update p_t
                for i in range(m):
                    p[(t+1)%dual_gap_freq,i,:], w_sum[i], w_square_sum[i] = \
                        SMD_p_Tree(c_vec, s_vec, r_vec, b_vec, beta,x[t%dual_gap_freq,:], p_tree_list[i], p[t%dual_gap_freq,i,:], coin_p_list[t % sample_freq,:],\
                              i,ss_p_list[i][t], delta,rho, RHS, \
                          emp_dist_value_copy, w_sum[i], w_square_sum[i])
                    # update addi and multi
                    multi[(t + 1) % dual_gap_freq, i] = p_tree_list[i].multi
                    addi[(t + 1) % dual_gap_freq, i] = p_tree_list[i].addi

                """
    
                Duality Gap Termination Condition(Implement when dual_flag != 0)
    
                """

                # Update bar_x and bar_p
                if dual_gap_cal == 0 and (t + 2) % dual_gap_freq == 0:
                    dual_gap_tic = time.time()
                    x_ws, ss_sum_x = bar_calculator_x(x_ws, x, dual_gap_freq, \
                                                      ss_x_list[
                                                      iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                                      ss_sum_x)
                    # update p with multi and addi, Optimize using broadcasting and in-place memory use.
                    last_p = p[-1, :, :].copy()
                    multi_expanded = np.expand_dims(multi, axis=-1)
                    addi_expanded = np.expand_dims(addi, axis=-1)
                    p *= multi_expanded  # In-place multiplication
                    p += addi_expanded  # In-place addition
                    actual_p = p
                    p_ws, ss_sum_p = bar_calculator_p(p_ws, actual_p, dual_gap_freq, \
                                                      ss_p_list[0][
                                                      iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                                      ss_sum_p)
                    dual_gap_toc = time.time()
                    dual_gap_time += dual_gap_toc - dual_gap_tic
                    # roll back the last p to the base p
                    # p[-1,:,:] is the last updated p value.
                    p[-1, :, :] = last_p
                    print('dual_gap_time:', dual_gap_time)
                # Calculate Dual gap
                if dual_gap_cal and (t + 2) % dual_gap_freq == 0:
                    dual_gap_tic = time.time()
                    x_ws, ss_sum_x, p_ws, ss_sum_p, sup_val, inf_val, diff = dual_gap_calculator(x_ws,x, dual_gap_freq,ss_x_list[iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                        ss_sum_x,p_ws,p, ss_p_list[0][iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq], ss_sum_p, multi, addi,
                        c_vec,s_vec,r_vec,b_vec,beta,emp_dist_value_copy,opt_model,var_x,var_y,var_tau,var_z,var_t,constr_list,rho,delta,alpha_tol,RHS)
                    dual_gap.append(diff)
                    dual_gap_toc = time.time()
                    dual_gap_time += dual_gap_toc - dual_gap_tic
                    print('dual_gap_time:', dual_gap_time)                



                    if print_option:
                        print("%s-st iteration duality gap:" % (t + 1), diff)
                        print("Dual Gap Calculation Time %s" % (dual_gap_time))
                        # print("x_bar:", x_bar)
                        # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
                        # print("p_new[0,0,0:]:", p_new[0,0,:])
                        # print("X Coeff:", temp_coeff)
                        print("sup_val:", sup_val)
                        print("inf_val:", inf_val)
                        if K_grad < n:
                            print("w_sum:", w_sum)


                #If K_test_flag == 1, we don't use duality gap termination condition
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

                """"
    
                At the very last iteration. Calculate the duality gap to verify our termination condition.
                Raise error if dual gap > obj_tol
    
                """

                if dual_gap_cal and t == T - 1 and print_option:
                    #array option output
                    real_t = T
                    dual_gap_tic = time.time()
                    x_ws, ss_sum_x, p_ws, ss_sum_p, sup_val, inf_val, diff = dual_gap_calculator(x_ws,x, dual_gap_freq,ss_x_list[iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                        ss_sum_x,p_ws,p, ss_p_list[0][iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq], ss_sum_p, multi, addi,
                        c_vec,s_vec,r_vec,b_vec,beta,emp_dist_value_copy,opt_model,var_x,var_y,var_tau,var_z,var_t,constr_list,rho,delta,alpha_tol,RHS)
                    dual_gap.append(diff)
                    dual_gap_toc = time.time()
                    dual_gap_time += (dual_gap_toc - dual_gap_tic)


                    print("%s-st iteration duality gap:" % (t + 1), diff)
                    if diff > obj_tol:
                        raise ValueError("Dual Gap greater than Obj_Tol.")
                    print("sup_val:", sup_val)
                    print("inf_val:", inf_val)
                    if K_grad < n:
                        print("w_sum:", w_sum)
                    else:
                        print('w_sum:', np.sum(p[t + 1], axis=1))

            if dual_gap_cal == 0:
                dual_gap_tic = time.time()
                # Calculate the duality gap at the last iteration if dual_gap_cal == 0
                x_ws, ss_sum_x, p_ws, ss_sum_p, sup_val, inf_val, diff = dual_gap_calculator(x_ws,x, dual_gap_freq,ss_x_list[iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                    ss_sum_x,p_ws,p, ss_p_list[0][iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq], ss_sum_p, multi, addi,
                    c_vec,s_vec,r_vec,b_vec,beta,emp_dist_value_copy,opt_model,var_x,var_y,var_tau,var_z,var_t,constr_list,rho,delta,alpha_tol,RHS)
                dual_gap.append(diff)
                print("%s-st iteration duality gap:" % T, diff)
                dual_gap_toc = time.time()
                dual_gap_time += dual_gap_toc - dual_gap_tic
                print('duality gap computation time:', dual_gap_time)
                if diff < obj_tol / 2:
                    solved_flag = 1

            dual_gap_list.append(dual_gap)
            real_T_list.append(real_t)
            iter_toc = time.time()
            total_solved_time += (iter_toc - iter_tic)
            total_dual_gap_time += dual_gap_time

            if break_flag:
                continue

        obj_val = (opt_up + opt_low) / 2

        print('Out of %s bisection iteration %s terminated early' % (bisection_count, early_term_count))
        print('Average iteration:', mean(real_T_list))
        print(f'total solved time with {bisection_count} bisection: {total_solved_time}')
        print(f'total solved time with {bisection_count} bisection without dual calculation time: {total_solved_time-total_dual_gap_time}')
            
        print('==========================================')

    stat = Statistics(n, m, K, K_grad, ss_type, obj_val, dual_gap_list,
                      iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_flag_count, 0, 0,
                      solved_flag=solved_flag, dual_gap_time=total_dual_gap_time)
    return stat

