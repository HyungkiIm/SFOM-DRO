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
from UBRegret import R_dim_SMD, R_const_SMD
from utils import *
from RBTree import Node, RedBlackTree


#Todo
"""
1. Go through Full_alpha function
2. Saving p value in tree structure? --> This is the most important thing.
3. Update dual_gap_freq to fraction. like 1/20.. --> Complete this today. 
5. Frequent update of x and p and x_bar and p_bar --> Complete this today. 
"""

"""
Here we use tree structure to save p values. We do not calculate duality gap here. (But we can actually implement it.)
Later change this code to save multiple iteration of p.
"""




"""
Define Stochastic Mirror Descent function for x
K: Sample size of our gradient 
m: Number of constraints
prox: prox function
p:current uncertainty parameter matrix, matrix form is m x J x L x N
alpha: step-size
x: current x
RHS: RHS of the constraint, for the first constraint(i=0) RHS is objective value that we are 
currently bisecting.
"""


#Here K_samples and K_grad_samples are list
def SMD_x_Tree(K, K_grad, x, step_alpha, K_samples, K_grad_samples, RHS, emp_dist_value, w_sum):
    """
    When Prox function is entropy function
    If we use chi-square as a f-divergence, we need to relax our uncertainty set according to
    Duchi's paper
    """
    # Get the value of m,J,L,n
    m, J, L, n = emp_dist_value.shape
    g_t = np.zeros([J, L])  # subgradient estimator
    F_hat = []

    """
    i_hat calculation
    """

    """
    Compute hat(F)^{i,K} for each constraint i
    hat(F)^{i,K} = 1/K* sum_{j=1}^K F^i(x_t,z^{i,I_j^i})
    """
    for i in range(m):
        temp = 0
        temp = w_sum[i] * np.sum(np.multiply(np.sum(emp_dist_value[i, :, :, :][:, :, K_samples[i]], \
                                                     axis=2), x))
        temp = temp / K - RHS[i]
        F_hat.append(temp)
    # get the index of max value in list F_hat
    i_hat = F_hat.index(max(F_hat))

    """
    g_t calculation
    """
    for k_idx in range(K_grad):
        g_t += w_sum[i_hat] * emp_dist_value[i_hat, :, :, K_grad_samples[k_idx]][0] / K_grad

    """
    x_{t+1} calculation
    """

    # Get next x (Projection Step with entropy prox function)
    theta = np.multiply(x, np.exp(-step_alpha*g_t))
    x_update = np.zeros([J, L])
    for j in range(J):
        x_update[j, :] = theta[j, :] / np.sum(theta[j, :])

    return x_update, np.array(F_hat)

def find_alpha(p_val, w_temp, n, delta, rho, w_sum, w_square_sum):
    """
    Input w_sum and w_square_sum are sum of p[t] and square sum of p[t].
    So, we need to perform an additional update on these values.

    """
    #Our input w_sum and w_square_sum is not updated. So we need to update before we proceed.

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

    if w_temp >= delta/n:
        if w_square_sum2 / 2 - w_sum2 / n + 1/(2*n) < rho/ n**2:
            alpha = 0
            return alpha
        else:
            alpha = 1 - math.sqrt(rho/(n**2 * (w_square_sum2 / 2 - w_sum2 / n + 1/(2*n))))
            return alpha

    alpha_thre = (delta/n - w_temp) / (1/n - w_temp)

    while True:

        alpha = (alpha_up + alpha_low) / 2


        if alpha >= alpha_thre: #I(alpha) = [n]
            w_sum = w_sum2
            w_square_sum = w_square_sum2
            I_alpha = I_alpha2
        else: #I(alpha) = [n]\I_t
            w_sum = w_sum1
            w_square_sum = w_square_sum1
            I_alpha = I_alpha1

        #Calculate g'(alpha)
        g_alpha = w_square_sum / 2 - w_sum / n + I_alpha * ((1 - alpha) ** 2 - (1 - delta) ** 2) / (2 * (n ** 2) * \
                (1 - alpha) ** 2) + (n * (1 - delta) ** 2 - 2 * rho) / (2 * (n ** 2) * (1 - alpha) ** 2)

        #Update the interval according to the g'(alpha) value
        if g_alpha < 0:
            alpha_up = alpha
        else:
            alpha_low = alpha

        #termination condition
        if alpha_low > alpha_thre: #I(alpha) = [n]
            if w_square_sum2 /2 - w_sum2/n + 1/(2*n) < rho/n**2:
                alpha = 0
                return alpha
            else:
                alpha = 1 - math.sqrt(rho/(n**2 * (w_square_sum2 / 2 - w_sum2 / n + 1/(2*n))))
                #raise TypeError('Returning case 1')
                return alpha
        elif alpha_up <= alpha_thre:
            if w_square_sum1 / 2 - w_sum1 / n + (n-1)/(2*n**2) <= rho/n**2 - (1-delta)**2 / (2*n**2):
                alpha = 0
                return alpha
            else:
                alpha = 1 - math.sqrt((rho/n**2 - (1-delta)**2 / (2*n**2))/(w_square_sum1 / 2 - w_sum1 / n + (n-1)/(2*n**2)))
                #raise TypeError('Returning case 2')
                return alpha


"""
Construct A_i for each i
We do not utilize tree structure here. 
prox function follows divergence of our problem
Here i is a index of constraint
"""


# In[10]:

#Now the input p is a tree
def SMD_p_Tree(x, p,p_arr, step_alpha, delta, rho,emp_dist_value, w_sum, w_square_sum):
    # for each constraint and JL sample one index I_t ~ p_t^i
    # If we are using chi-square as our f-divergence, we need to relax our uncertainty set.
    J, L, n = emp_dist_value.shape

    coin_list = np.random.uniform(size = math.ceil(2 * np.log2(n+1))).reshape((1,math.ceil(2 * np.log2(n+1))))

    I_t = p.random_sample(coin_list)[0]

    old_pval = p.multi * p_arr[I_t] + p.addi

    grad_val = np.sum(np.multiply(x, emp_dist_value[:, :, I_t])) * w_sum / old_pval
    #update p_t to w_t
    w_temp = old_pval + step_alpha * grad_val
    # Projection to our chi-square uncertainty set
    # We are not using tree structure here.
    # Note that g'(alpha) is a decreasing function of alpha
    # Input w_sum and w_square_sum are sum of p[t] and square sum of p[t].
    alpha = find_alpha(old_pval, w_temp, n, delta, rho, w_sum, w_square_sum)

    #Update multi and addi.
    p.multi = p.multi * (1-alpha)
    p.addi = p.addi * (1-alpha) + alpha/n

    if w_temp < delta / n:

        p.delete_node(p_arr[I_t], I_t)
        p.insert((delta/n- p.addi) / p.multi, I_t)
        w_square_sum = (1 - alpha) ** 2 * (w_square_sum - old_pval ** 2) + 2 * (1 - alpha) * alpha * \
                       (w_sum - old_pval) / n + (n - 1) * alpha ** 2 / n ** 2 + delta ** 2 / n ** 2
        w_sum = (1 - alpha) * (w_sum - old_pval) + (n - 1) * alpha / n + delta / n

        p_arr[I_t] = (delta/n- p.addi) / p.multi

    else:  # p_new[I_t] > delta/n // We do not need to update tree. Updating multi and addi is sufficient. 

        w_sum = w_sum + step_alpha * grad_val
        w_square_sum = w_square_sum - old_pval ** 2 + w_temp ** 2 #Recheck this part later.
        w_square_sum = (1 - alpha) ** 2 * w_square_sum + 2 * alpha * (1 - alpha) * w_sum / n + alpha ** 2 / n
        w_sum = (1 - alpha) * w_sum + alpha

    """
    #Check whether our modification is correct.
    if np.sum(p_new[j,l,:])!= w_sum[j,l]:
        print(np.sum(p_new[j,l,:]))
        print(w_sum[j,l])
        raise TypeError("w_sum not equal")
    if np.sum(p_new[j,l,:]**2)!= w_square_sum[j,l]:
        print(np.sum(p_new[j,l,:]**2))
        print(w_square_sum[j,l])
        raise TypeError("w_square_sum not equal")
    """

    return  w_sum, w_square_sum


def DRO_SMD_Tree(x_0, p_0, emp_dist_value, K, K_grad, delta, rho, opt_low, opt_up, obj_tol,
               RHS,ss_type, C_K, T_cap, print_option=1, K_test_flag=0, min_flag=1):
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

    stoc_factor = 1  # We need to adjust this part later Sep-20

    # G is bound of norm of gradient(\nabla) F_k^i(x) for all k \in [n] and i \in [m] and x \in X
    G = np.absolute(emp_dist_value).max()  # Also, need to change this funciton add C2
    # M is an list of bound of |F_k^i(x)| for each i \in [m]
    absolute_max = np.absolute(emp_dist_value).max(axis=2)
    sum_over_J = np.sum(absolute_max, axis=1)
    M = sum_over_J.max(axis=1)

    # Calculate T and our stepsize
    if ss_type == 'constant':
        T, R_x, R_p, ss_x, ss_p = R_const_SMD(J, L, n, G, M, delta, rho, obj_tol, C_G, C_K, stoc_factor)
        if K_test_flag:
            print("Max Iteration:", T_cap)
            print('alg_type:SGD with i_hat')
            T = T_cap
            obj_tol = 1e-7
            ss_x_list = ss_x * np.ones(T + 1)
            ss_p_list = []
            for i in range(m):
                ss_p_list.append(ss_p[i] * np.ones(T + 1))
        else:
            print("Max Iteration:", T)
            print('alg_type:SGD with i_hat')
            ss_x_list = ss_x * np.ones(T + 1)
            ss_p_list = []
            for i in range(m):
                ss_p_list.append(ss_p[i] * np.ones(T + 1))



    elif ss_type == 'diminish':
        T, R_x, R_p, c_x, c_p = R_dim_SMD(J, L, n, G, M, delta, rho, obj_tol, C_G, C_K,
                                          stoc_factor)
        if K_test_flag:
            print("Max Iteration:", T_cap)
            print('alg_type:SGD with i_hat')
            T = T_cap
            ss_x_list = c_x * (np.sqrt(1 / (np.arange(T + 1) + 1)))
            obj_tol = 1e-7
            ss_p_list = []
            for i in range(m):
                ss_p_list.append(c_p[i] * (np.sqrt(1 / (np.arange(T + 1) + 1))))

        else:
            print("Max Iteration:", T)
            print('alg_type:SGD with i_hat')
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



    sample_freq = 10 ** 3  # We can move this to parameter.
    sanity_check = 0
    sanity_freq = 1000


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
                print('alg_type:SGD with i_hat')
                print('---------------------------')
                print("current step objective value:", obj_val)

            #If dual_gap_freq == 0, then we do not save previous output
            #Todo
            #We are saving x_t only. Exclude this step when it is needed.

            x = np.empty([J,L])
            x = x_0
            p_tree_list = []

            # bst.construct intial weight equal to p_0
            for i in range(m):
                bst = RedBlackTree()
                bst.construct_tree(n)
                p_tree_list.append(bst)

            #This array tracks weight of each node in tree. To get the actual value of p we should multi*p +addi.
            p_arr = p_0.copy()
            # We use this array to save hat_f at every iteration. We use this value to calculate vartheta.
            # Currently we just exclude x_{T+1}.
            f_hat_arr = np.zeros([T,m])



            # Get x and p for T iterations
            tic = time.time()

            w_sum = np.zeros(m)
            w_sum[:] = np.nan
            w_square_sum = np.zeros(m)
            w_square_sum[:] = np.nan

            for i in range(m):
                w_sum[i] = np.sum(p_0[i, :])
                w_square_sum[i] = np.sum(p_0[i, :] ** 2)

            dual_gap = []  # List that contains duality gap in this bisection

            # if dual_gap_freq == 0 : #We also need to update this value periodically.
            #
            #     for i in range(m):
            #         f_val[i] += np.sum(np.multiply(np.tensordot(p_old[i, :],emp_dist_value_copy[i, :, :, :], \
            #                                                                  axes=(0, 2)), x_list[0])) * ss_x_list[0]

            sampling_time = 0
            for_loop_time = 0
            for t in range(T):

                # Now our step-size is not uniform. Make sure to change x_bar and p_bar according to our step size
                # Create samples

                if t % sample_freq == 0 and print_option:

                    coin_K_arr = np.random.uniform(size = [sample_freq, m, K, math.ceil(2 * np.log2(n+1))])
                    coin_grad_arr = np.random.uniform(size = [sample_freq,m,K_grad,math.ceil(2 * np.log2(n+1))])

                    toc = time.time()
                    print('=========================================')
                    print('%s-st iteration start time:' % (t + 1), toc - tic)
                    print('%s-st sampling time:' % (t + 1), sampling_time)
                    print('%s-st for loop time:' % (t + 1), for_loop_time)
                #We do not need information of p_t here. Only cumdist and w_sum is needed.


                #Implement random sampling using bst.random_sample
                samples_list = []
                grad_samples_list = []
                sample_tic = time.time()
                for i in range(m):
                    sample_idx = p_tree_list[i].random_sample(coin_K_arr[t % sample_freq, i, :, :])
                    samples_list.append(sample_idx)
                    sample_idx =  p_tree_list[i].random_sample(coin_grad_arr[t % sample_freq, i, :, :])
                    grad_samples_list.append(sample_idx)
                sample_toc = time.time()
                sampling_time += sample_toc - sample_tic

                #Update x
                x_old = x.copy()
                x, f_hat_arr[t,:] = SMD_x_Tree(K, K_grad, x, ss_x_list[t],samples_list, grad_samples_list,RHS, \
                               emp_dist_value_copy, w_sum)


                #Update p
                for i in range(m):
                    w_sum[i], w_square_sum[i] = SMD_p_Tree(x_old, p_tree_list[i],p_arr[i], \
                        ss_p_list[i][t], delta,rho, emp_dist_value_copy[i,:,:,:], w_sum[i], w_square_sum[i])


                    # f_val[i] += np.sum(np.multiply(np.tensordot(p_old[i,:],emp_dist_value_copy[i,:,:,:],\
                    #                       axes = (0,2)),x[t+1,:,:])) * ss_x_list[t]



                #Sanity Check
                # if dual_gap_freq == 0 and sanity_check:
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
                #         if K_grad < n:
                #             print("w_sum:", w_sum)
                #         else:
                #             print('w_sum:',
                #                   np.sum(p[t + 1], axis=1))  # We need to turn this print option later.



            f_val = np.zeros(m)

            for i in range(m):
                f_val[i] = np.average(f_hat_arr, weights = ss_x_list[:T+1])

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
    #         print('alg_type:SGD with i_hat')
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


    #dual_gap_list, p_bar, p, should be empty list or value if dual_gap_freq == 0.
    if dual_gap_freq == 0:
        p_bar = None

    stat = Statistics(n, m, K, K_grad, ss_type, x_bar, p_bar, obj_val, x, p, dual_gap_list,
                      iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_flag_count)

    # Update the last objective value

    # obj_val = (-1 + 2*min_flag) * obj_val

    return stat