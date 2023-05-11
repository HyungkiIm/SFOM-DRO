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
from copy import deepcopy


#Todo
"""
1. Go through Full_alpha function
2. Saving p value in tree structure? --> This is the most important thing.
3. Update dual_gap_freq to fraction. like 1/20.. --> Complete this today. 
5. Frequent update of x and p and x_bar and p_bar --> Complete this today. 
"""

"""
Here, we use following structure to save previous solutions and update p_bar and x_bar.
We have 4 options for the DRO_SMD algorithm. This code is incomplete. 
x = np.empty([dual_gap_freq,J,L])
x[0,:,:] = x_0
p = np.empty([dual_gap_freq, m, n])
p[0,:,:] = p_0

"""



"""

Update Notes:
Oct-11

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


def SMD_x(K, K_grad, x, step_alpha, samples_list, grad_samples_list, RHS, emp_dist_value, w_sum, cum_dist):
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
    K_samples = np.zeros([m, K])

    # Pick K samples from each constraint, and j and l.
    for i in range(m):
        K_samples[i, :] = np.digitize(samples_list[i, :], cum_dist[i, :])

    K_samples = np.where(K_samples >= n, n - 1, K_samples)

    """
    Compute hat(F)^{i,K} for each constraint i
    hat(F)^{i,K} = 1/K* sum_{j=1}^K F^i(x_t,z^{i,I_j^i})
    """


    for i in range(m):
        temp = 0
        temp = w_sum[i] * np.sum(np.multiply(np.sum(emp_dist_value[i, :, :, :][:, :, K_samples[i, :].astype(int)], \
                                                     axis=2), x))
        temp = temp / K - RHS[i] * w_sum[i]
        F_hat.append(temp)
    # get the index of max value in list F_hat
    i_hat = F_hat.index(max(F_hat))

    """
    g_t calculation
    """


    K_grad_samples = np.zeros(K_grad)
    K_grad_samples[:] = np.digitize(grad_samples_list, cum_dist[i_hat, :])
    K_grad_samples = np.where(K_grad_samples >= n, n - 1, K_grad_samples)
    for k_idx in range(K_grad):
        g_t += w_sum[i_hat] * emp_dist_value[i_hat, :, :, int(K_grad_samples[k_idx])] / K_grad

    """
    x_{t+1} calculation
    """

    # Get next x (Projection Step with entropy prox function)
    theta = np.multiply(x, np.exp(-step_alpha*g_t))
    x_update = np.zeros([J, L])
    for j in range(J):
        x_update[j, :] = theta[j, :] / np.sum(theta[j, :])

    return x_update, np.array(F_hat)

def SMD_x_K_test(K, K_grad,C_K,obj_tol, x, p, step_alpha, samples_list, grad_samples_list, RHS, emp_dist_value, w_sum, cum_dist):
    """
    When Prox function is entropy function
    If we use chi-square as a f-divergence, we need to relax our uncertainty set according to
    Duchi's paper
    """

    eps = C_K * obj_tol

    # Get the value of m,J,L,n
    m, J, L, n = emp_dist_value.shape
    g_t = np.zeros([J, L])  # subgradient estimator
    F_hat = []
    F_hat_real = []
    i_hat_flag = 0

    """
    i_hat calculation
    """
    K_samples = np.zeros([m, K])

    # Pick K samples from each constraint, and j and l.
    for i in range(m):
        K_samples[i, :] = np.digitize(samples_list[i, :], cum_dist[i, :])

    K_samples = np.where(K_samples >= n, n - 1, K_samples)

    """
    Compute hat(F)^{i,K} for each constraint i
    hat(F)^{i,K} = 1/K* sum_{j=1}^K F^i(x_t,z^{i,I_j^i})
    """


    for i in range(m):
        temp = 0
        temp = w_sum[i] * np.sum(np.multiply(np.sum(emp_dist_value[i, :, :, :][:, :, K_samples[i, :].astype(int)], \
                                                     axis=2), x))
        temp = temp / K - RHS[i] * w_sum[i]
        F_hat.append(temp)
    # get the index of max value in list F_hat
    i_hat = F_hat.index(max(F_hat))

    for i in range(m):
        temp = 0
        temp = np.sum(np.multiply(np.tensordot(p[i, :], emp_dist_value[i, :, :, :], axes=(0, 2)), x))
        temp = temp - RHS[i] * w_sum[i]
        F_hat_real.append(temp)

    i_star = F_hat_real.index(max(F_hat_real))

    """
    i_hat and i_star comparison
    """

    if abs(F_hat_real[i_star] -F_hat[i_hat]) <= eps:
        i_hat_flag = 1

    """
    g_t calculation
    """


    K_grad_samples = np.zeros(K_grad)
    K_grad_samples[:] = np.digitize(grad_samples_list, cum_dist[i_hat, :])
    K_grad_samples = np.where(K_grad_samples >= n, n - 1, K_grad_samples)
    for k_idx in range(K_grad):
        g_t += w_sum[i_hat] * emp_dist_value[i_hat, :, :, int(K_grad_samples[k_idx])] / K_grad

    """
    x_{t+1} calculation
    """

    # Get next x (Projection Step with entropy prox function)
    theta = np.multiply(x, np.exp(-step_alpha*g_t))
    x_update = np.zeros([J, L])
    for j in range(J):
        x_update[j, :] = theta[j, :] / np.sum(theta[j, :])

    return x_update, np.array(F_hat), i_hat_flag



def find_alpha(p_val, w_t, I_t, n, delta, rho, w_sum, w_square_sum):
    """
    Input w_sum and w_square_sum are sum of p[t] and square sum of p[t].
    So, we need to perform an additional update on these values.

    """
    #Our input w_sum and w_square_sum is not updated. So we need to update before we proceed.

    alpha_up = 1
    alpha_low = 0
    g_alpha = 1

    # values when w_t[I_t]>= w_threshold, I(alpha) = [n]
    w_sum2 = w_sum - p_val + w_t[I_t]
    w_square_sum2 = w_square_sum - p_val ** 2 + w_t[I_t] ** 2
    I_alpha2 = n

    # values when w_t[I_t]< w_threshold, I(alpha) = [n]\[I_t]
    w_sum1 = w_sum - p_val
    w_square_sum1 = w_square_sum - p_val ** 2
    I_alpha1 = n - 1

    if w_t[I_t] >= delta/n:
        if w_square_sum2 / 2 - w_sum2 / n + 1/(2*n) < rho/ n**2:
            alpha = 0
            return alpha
        else:
            alpha = 1 - math.sqrt(rho/(n**2 * (w_square_sum2 / 2 - w_sum2 / n + 1/(2*n))))
            return alpha

    alpha_thre = (delta/n - w_t[I_t]) / (1/n - w_t[I_t])

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


def SMD_p(x, p, i, step_alpha, delta, rho, RHS, emp_dist_value, w_sum, w_square_sum, cum_dist):
    # for each constraint and JL sample one index I_t ~ p_t^i
    # If we are using chi-square as our f-divergence, we need to relax our uncertainty set.
    _, J, L, n = emp_dist_value.shape

    #Sample an index
    random_num = np.random.rand()
    I_t = np.digitize(random_num, cum_dist)

    if I_t >=n:
        I_t -=1

    grad_val = (np.sum(np.multiply(x, emp_dist_value[i, :, :, I_t]))-RHS[i]) * w_sum / p[I_t]
    old_pval = deepcopy(p[I_t])
    #update p_t to w_t
    p[I_t] += step_alpha * grad_val
    w_temp = p[I_t]
    # Projection to our chi-square uncertainty set
    # We are not using tree structure here.
    # Note that g'(alpha) is a decreasing function of alpha
    # Input w_sum and w_square_sum are sum of p[t] and square sum of p[t].
    alpha = find_alpha(old_pval, p, I_t, n, delta, rho, w_sum, w_square_sum)

    # Update p value according to alpha that we find above
    # For i != I_t, p_{t+1} = (1-alpha)w_i + alpha/n, Can we reduce this process to O(log n)
    p *= (1 - alpha)
    p += alpha / n

    #Update cumulative distribution
    v1 = alpha * (np.arange(n) + 1) / n # We need to work on this
    # For i = I_t, we should first take a look at w_{t,I_t}
    # Also update cumulative distribution vector here.

    if p[I_t] < delta / n:
        gamma = (delta - alpha) / n - (1 - alpha) * old_pval
        temp1 = np.zeros(I_t)
        temp2 = gamma * np.ones(n - I_t)

        v2 = np.concatenate((temp1, temp2))
        cum_dist *= (1 - alpha) * w_sum
        cum_dist += (v1+v2)
        p[I_t] = delta / n
        w_square_sum = (1 - alpha) ** 2 * (w_square_sum - old_pval ** 2) + 2 * (1 - alpha) * alpha * \
                       (w_sum - old_pval) / n + (n - 1) * alpha ** 2 / n ** 2 + delta ** 2 / n ** 2
        w_sum = (1 - alpha) * (w_sum - old_pval) + (n - 1) * alpha / n + delta / n
        cum_dist /= w_sum
    # check whether the two value is correct.
    else:  # p_new[I_t] > delta/n
        cum_dist *= (1 - alpha) * w_sum
        temp1 = np.zeros(I_t)
        temp2 = step_alpha * grad_val * np.ones(n - I_t)
        v2 = np.concatenate((temp1, temp2))
        cum_dist += v1 + v2

        w_sum = w_sum + step_alpha * grad_val
        w_square_sum = w_square_sum - old_pval ** 2 + w_temp ** 2 #Recheck this part later.
        w_square_sum = (1 - alpha) ** 2 * w_square_sum + 2 * alpha * (1 - alpha) * w_sum / n + alpha ** 2 / n
        w_sum = (1 - alpha) * w_sum + alpha
        cum_dist /= w_sum


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

    return p, w_sum, w_square_sum, cum_dist


def DRO_SMD(x_0, p_0, emp_dist_value, K, K_grad, delta, rho, alpha_tol, opt_low, opt_up, obj_tol,
        RHS,ss_type, C_K, dual_gap_option, dual_gap_freq, T_cap, print_option=1, K_test_flag=0, min_flag=1,feas_opt = 0):
    # Change data according to min_flag.
    """
    If dual_gap_option == 0: then don't calculate dual_gap and use absolute dual_gap_freq.
    If dual_gap_option == 1: then don't calculate dual_gap and use relative dual_gap_freq.
    If dual_gap_option == 2: then calculate dual_gap and use absolute dual_gap_freq.
    If dual_gap_option == 3: then calculate dual_gap and use relative dual_gap_freq.
    """

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



    sample_freq = 10 ** 4  # We can move this to parameter.
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


    #Check Whether hat_vartheta is approximating well.
    vartheta_flag = 0
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
                print('alg_type:SGD with i_hat')
                print('---------------------------')
                print("current step objective value:", obj_val)

            #If dual_gap_freq == 0, then we do not save previous output
            #Todo
            #We are saving x_t only. Exclude this step when it is needed.
            x = np.empty([dual_gap_freq,J,L])
            x[0,:,:] = x_0
            p = np.empty([dual_gap_freq, m, n])
            p[0,:,:] = p_0

            hat_f_val = np.zeros([dual_gap_freq,m])
            hat_f_val_ws = np.zeros(m)


            #Variables that is needed to update periodically.
            iter_count = 0
            ss_sum_x = 0
            ss_sum_p = 0 #This does not have to be np.zeros([m])
            x_ws = np.zeros([J,L])
            p_ws = np.zeros([m,n])
            if vartheta_flag:
                real_f_val = np.zeros([dual_gap_freq, m])
                real_f_val_ws = np.zeros(m)


            # Get x and p for T iterations
            tic = time.time()

            w_sum = np.zeros(m)
            w_sum[:] = np.nan
            w_square_sum = np.zeros(m)
            w_square_sum[:] = np.nan
            cum_dist = np.zeros([m, n])
            cum_dist[:] = np.nan

            for i in range(m):
                w_sum[i] = np.sum(p_0[i, :])
                w_square_sum[i] = np.sum(p_0[i, :] ** 2)
                cum_dist[i, :] = np.cumsum(p_0[i, :])

            dual_gap = []  # List that contains duality gap in this bisection

            # if dual_gap_freq == 0 : #We also need to update this value periodically.
            #
            #     for i in range(m):
            #         f_val[i] += np.sum(np.multiply(np.tensordot(p_old[i, :],emp_dist_value_copy[i, :, :, :], \
            #                                                                  axes=(0, 2)), x_list[0])) * ss_x_list[0]


            for t in range(T):

                # Now our step-size is not uniform. Make sure to change x_bar and p_bar according to our step size
                # Create samples
                if t % sample_freq == 0:
                    temp_samples_list = np.random.rand(m, K, sample_freq)
                    grad_samples_list = np.random.rand(K_grad, sample_freq)


                #hat_vartheta sanity check

                if vartheta_flag:
                    real_F_hat = []
                    for i in range(m):
                        temp = 0
                        temp = np.sum(np.multiply(np.tensordot(p[t%dual_gap_freq,i, :], emp_dist_value_copy[i, :, :, :],\
                            axes=(0, 2)), x[t%dual_gap_freq,:,:]))
                        temp = temp - RHS[i]
                        real_F_hat.append(temp)
                    real_f_val[t%dual_gap_freq,:] = np.array(real_F_hat)




                if print_option and (t+2)%dual_gap_freq ==0:
                    toc = time.time()
                    print('=========================================')
                    print('%s-st iteration start time:' % (t + 1), toc - tic)
                #We do not need information of p_t here. Only cumdist and w_sum is needed.
                x[(t+1)%dual_gap_freq,:,:],hat_f_val[t%dual_gap_freq,:] = SMD_x(K, K_grad, x[t%dual_gap_freq,:,:],\
                    ss_x_list[t], temp_samples_list[:, :, t % sample_freq], grad_samples_list[:,t % sample_freq],RHS, \
                        emp_dist_value_copy, w_sum, cum_dist)

                for i in range(m):
                    p[(t+1)%dual_gap_freq,i,:], w_sum[i], w_square_sum[i], cum_dist[i, :] = \
                        SMD_p(x[t%dual_gap_freq,:,:], p[t%dual_gap_freq,i,:], i,ss_p_list[i][t], delta,rho, RHS, \
                          emp_dist_value_copy, w_sum[i], w_square_sum[i], cum_dist[i, :])

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




                """
    
                Duality Gap Termination Condition(Implement when dual_flag != 0)
    
                """
                # Calculate Dual gap
                if dual_gap_cal and (t + 2) % dual_gap_freq == 0:
                    dual_gap_tic = time.time()
                    x_ws, ss_sum_x = bar_calculator_temp(x_ws, x, dual_gap_freq,\
                        ss_x_list[iter_count * dual_gap_freq:(iter_count+1) * dual_gap_freq],ss_sum_x)
                    p_ws, ss_sum_p = bar_calculator_temp(p_ws,p, dual_gap_freq,\
                                ss_p_list[0][iter_count * dual_gap_freq:(iter_count+1) * dual_gap_freq],ss_sum_p)
                    sup_val = sup_pi(x_ws/ss_sum_x, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
                    inf_val = inf_pi(p_ws/ss_sum_p, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
                    diff = sup_val - inf_val
                    dual_gap.append(diff)
                    dual_gap_toc = time.time()

                if dual_gap_cal and (t+2) % dual_gap_freq == 0 and print_option:
                    print("%s-st iteration duality gap:" % (t + 1), diff)
                    print("Dual Gap Calculation Time %s" %(dual_gap_toc - dual_gap_tic))
                    # print("x_bar:", x_bar)
                    # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
                    # print("p_new[0,0,0:]:", p_new[0,0,:])
                    # print("X Coeff:", temp_coeff)
                    pi = pi_val(x_ws/ss_sum_x, p_ws/ss_sum_p, emp_dist_value_copy, RHS)
                    print("sup_val:", sup_val)
                    print("inf_val:", inf_val)
                    print('pi_val:', pi)
                    if K_grad < n:
                        print("w_sum:", w_sum)
                        p_temp = p[(t + 1) % dual_gap_freq, :, :]
                        print('divergence:', np.sum((p_temp - 1 / n) ** 2, axis=1) * n / 2, 'rho/n=', rho/n)

                """
    
                Update hat_f_val_ws every dual_gap_freq iteration.
    
                """
                # Whether dual_gap_cal ==0 or not, we calculate hat_f_val_ws. Also, this update comes later than dual
                # gap calculation, so we increase our iter_count here.
                if (t + 1) % dual_gap_freq == 0:
                    hat_f_val_ws += np.tensordot(hat_f_val, ss_x_list[iter_count * dual_gap_freq: \
                                                                      (iter_count + 1) * dual_gap_freq], axes=(0, 0))

                    if vartheta_flag:

                        real_f_val_ws +=np.tensordot(real_f_val, ss_x_list[iter_count * dual_gap_freq: \
                                                                          (iter_count + 1) * dual_gap_freq], axes=(0, 0))
                    iter_count += 1


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

                """"
    
                At the very last iteration. Calculate the duality gap to verify our termination condition.
                Raise error if dual gap > obj_tol
    
                """

                if dual_gap_cal and t == T - 1 and print_option:
                    #array option output
                    real_t = T
                    x_ws, ss_sum_x = bar_calculator_temp(x_ws, x, T+1 - iter_count*dual_gap_freq, \
                                    ss_x_list[iter_count * dual_gap_freq:],ss_sum_x)
                    p_ws, ss_sum_p = bar_calculator_temp(p_ws, p, T+1- iter_count*dual_gap_freq,\
                                                    ss_p_list[0][iter_count * dual_gap_freq:], ss_sum_p)
                    sup_val = sup_pi(x_ws / ss_sum_x, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
                    inf_val = inf_pi(p_ws / ss_sum_p, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
                    diff = sup_val - inf_val
                    dual_gap.append(diff)


                    print("%s-st iteration duality gap:" % (t + 1), diff)
                    if diff > obj_tol:
                        raise ValueError("Dual Gap greater than Obj_Tol.")
                    # print("x_bar:", x_bar)
                    # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
                    # print("p_new[0,0,0:]:", p_new[0,0,:])
                    # print("X Coeff:", temp_coeff)
                    print("sup_val:", sup_val)
                    print("inf_val:", inf_val)
                    if K_grad < n:
                        print("w_sum:", w_sum)
                    else:
                        print('w_sum:', np.sum(p[t + 1], axis=1))

            if dual_gap_cal == 0:
                real_t = T

            dual_gap_list.append(dual_gap)
            real_T_list.append(real_t)


            if break_flag:
                continue

            #Calculate the last hat_f_val_ws

            hat_f_val_ws += np.tensordot(hat_f_val[: T % dual_gap_freq], ss_x_list[iter_count * dual_gap_freq:T], axes=(0, 0))
            ss_sum_x = np.sum(ss_x_list[:T])
            hat_f_val_ws /= ss_sum_x

            if vartheta_flag:
                real_f_val_ws += np.tensordot(real_f_val[: T % dual_gap_freq], ss_x_list[iter_count * dual_gap_freq:T],
                                             axes=(0, 0))
                real_f_val_ws /= ss_sum_x

                print("==========Vartheta Comparison===========")
                print("hat_f_val_ws: %s" %hat_f_val_ws)
                print("real_f_val_ws: %s" %real_f_val_ws)
                print("difference with real_f_val_ws: %s" %(hat_f_val_ws - real_f_val_ws))
                if (hat_f_val_ws - real_f_val_ws).max() > 1e-3:
                    raise ValueError("hat_vartheta not approximating well. Increase Sample Size.")
                print("==========================================")
            hat_vartheta = hat_f_val_ws.max()
            # Now implement Bisection // We are using hat_vartheta. Our threshold value changed.

            if hat_vartheta > R_x + C_K * obj_tol:
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

    stat = Statistics(n, m, K, K_grad, ss_type, obj_val, dual_gap_list,
                      iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_flag_count)

    # Update the last objective value

    # obj_val = (-1 + 2*min_flag) * obj_val

    return stat

def DRO_SMD_K_test_iter(x_0, p_0, emp_dist_value, K, K_grad, delta, rho, alpha_tol, opt_low, opt_up, obj_tol,
        RHS,ss_type, C_K, dual_gap_option, dual_gap_freq, T_cap, print_option=1, min_flag=1):

    emp_dist_value_copy = emp_dist_value.copy()
    emp_dist_value_copy[0, :, :, :] = (-1 + 2 * min_flag) * emp_dist_value_copy[0, :, :, :]
    #RHS = np.ones(RHS.shape[0])
    RHS[0] = (-1 + 2 * min_flag) * RHS[0]
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
        print("Max Iteration:", T_cap)
        print('alg_type:SGD with i_hat')
        T = T_cap
        #obj_tol = 1e-7
        ss_x_list = ss_x * np.ones(T + 1)
        ss_p_list = []
        for i in range(m):
            ss_p_list.append(ss_p[i] * np.ones(T + 1))




    elif ss_type == 'diminish':
        T, R_x, R_p, c_x, c_p = R_dim_SMD(J, L, n, G, M, delta, rho, obj_tol, C_G, C_K,
                                          stoc_factor)
        print("Max Iteration:", T_cap)
        print('alg_type:SGD with i_hat')
        T = T_cap
        ss_x_list = c_x * (np.sqrt(1 / (np.arange(T + 1) + 1)))
        #obj_tol = 1e-7
        ss_p_list = []
        for i in range(m):
            ss_p_list.append(c_p[i] * (np.sqrt(1 / (np.arange(T + 1) + 1))))


    print('Rx: %s' %R_x)
    print('Rp: %s' %R_p)

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

    if dual_gap_option == 2 or dual_gap_option == 0:
        pass
    elif dual_gap_option == 3 or dual_gap_option == 1:
        dual_gap_freq = int(T * dual_gap_freq)

    # Set calculation option
    if dual_gap_option == 0 or dual_gap_option == 1:
        dual_gap_cal = 0
    else:
        dual_gap_cal = 1


    bisection_count += 1
    # Change our objective function
    obj_val = (opt_up + opt_low) / 2

    if print_option:
        print('---------------------------')
        print('%s-th bisection iteration' % bisection_count)
        print('alg_type:SGD with i_hat')
        print('---------------------------')
        print("current step objective value:", obj_val)

    x = np.empty([dual_gap_freq, J, L])
    x[0, :, :] = x_0
    p = np.empty([dual_gap_freq, m, n])
    p[0, :, :] = p_0

    hat_f_val = np.zeros([dual_gap_freq, m])
    hat_f_val_ws = np.zeros(m)

    # Variables that is needed to update periodically.
    iter_count = 0
    ss_sum_x = 0
    ss_sum_p = 0  # This does not have to be np.zeros([m])
    x_ws = np.zeros([J, L])
    p_ws = np.zeros([m, n])

    # Get x and p for T iterations
    tic = time.time()

    w_sum = np.zeros(m)
    w_sum[:] = np.nan
    w_square_sum = np.zeros(m)
    w_square_sum[:] = np.nan
    cum_dist = np.zeros([m, n])
    cum_dist[:] = np.nan

    for i in range(m):
        w_sum[i] = np.sum(p_0[i, :])
        w_square_sum[i] = np.sum(p_0[i, :] ** 2)
        cum_dist[i, :] = np.cumsum(p_0[i, :])

    dual_gap = []  # List that contains duality gap in this bisection

    sup_val = sup_pi(x_0, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
    inf_val = inf_pi(p_0, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
    diff = sup_val - inf_val
    dual_gap.append(diff)

    for t in range(T):

        # Now our step-size is not uniform. Make sure to change x_bar and p_bar according to our step size
        # Create samples
        if t % sample_freq == 0:
            temp_samples_list = np.random.rand(m, K, sample_freq)
            grad_samples_list = np.random.rand(K_grad, sample_freq)

        if print_option and (t + 2) % dual_gap_freq == 0:
            toc = time.time()
            print('=========================================')
            print('%s-st iteration start time:' % (t + 1), toc - tic)
        # We do not need information of p_t here. Only cumdist and w_sum is needed.
        x[(t + 1) % dual_gap_freq, :, :], hat_f_val[t % dual_gap_freq, :], i_hat_flag = SMD_x_K_test(K, K_grad,C_K,obj_tol,
                x[t % dual_gap_freq, :, :],p[t%dual_gap_freq,:,:] ,ss_x_list[t],temp_samples_list[:, :, t % sample_freq],
                       grad_samples_list[:,t % sample_freq], RHS, emp_dist_value_copy, w_sum, cum_dist)
        i_flag_count += i_hat_flag

        for i in range(m):
            p[(t + 1) % dual_gap_freq, i, :], w_sum[i], w_square_sum[i], cum_dist[i, :] = \
                SMD_p(x[t % dual_gap_freq, :, :], p[t % dual_gap_freq, i, :], i, ss_p_list[i][t], delta, rho, RHS,
                      emp_dist_value_copy, w_sum[i], w_square_sum[i], cum_dist[i, :])


        """

        Calculate Duality Gap

        """
        # Calculate Dual gap
        if dual_gap_cal and (t + 2) % dual_gap_freq == 0:
            x_ws, ss_sum_x = bar_calculator_temp(x_ws, x, dual_gap_freq, \
                                                 ss_x_list[
                                                 iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                                 ss_sum_x)
            p_ws, ss_sum_p = bar_calculator_temp(p_ws, p, dual_gap_freq, \
                                                 ss_p_list[0][
                                                 iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                                 ss_sum_p)
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
            if K_grad < n:
                print("w_sum:", w_sum)

        """

        Update hat_f_val_ws every dual_gap_freq iteration.

        """
        # Whether dual_gap_cal ==0 or not, we calculate hat_f_val_ws. Also, this update comes later than dual
        # gap calculation, so we increase our iter_count here.
        if (t + 1) % dual_gap_freq == 0:
            hat_f_val_ws += np.tensordot(hat_f_val, ss_x_list[iter_count * dual_gap_freq: \
                                                              (iter_count + 1) * dual_gap_freq], axes=(0, 0))
            iter_count += 1

    dual_gap_list.append(dual_gap)
    total_toc = time.time()
    total_solved_time = total_toc - total_tic

    SMD_stat = Statistics(n, m, K, K_grad, ss_type, obj_val, dual_gap_list,
                      iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_flag_count)

    return SMD_stat

# def DRO_SMD_K_test_time(x_0, p_0, emp_dist_value, K, K_grad, delta, rho, alpha_tol, opt_low, opt_up, obj_tol,
#         RHS,ss_type, C_K, dual_gap_option, dual_gap_freq, time_cap, time_freq, print_option=1, min_flag=1):
#     """
#     We implement K_test_time with this function. At every time_freq, we calculate the dual gap and record
#     #its value.
#     """
#     emp_dist_value_copy = emp_dist_value.copy()
#     emp_dist_value_copy[0, :, :, :] = (-1 + 2 * min_flag) * emp_dist_value_copy[0, :, :, :]
#
#     m, J, L, n = emp_dist_value.shape  # Parameters
#
#     # Calculate coefficients
#
#     C_G = 1 + math.sqrt(rho / n)
#
#     print('\n')
#     print('************************************************************************************')
#     print('*******************************Problem Description**********************************')
#     print(' ')
#     print('Number of constraints: %s, x dimension: %s ,Uncertainty Set Dimension: %s' % (m, J * L, n))
#     print(' ')
#     print('*************************************************************************************')
#     print('*************************************************************************************')
#
#     i_flag_count = 0
#     print('We are doing sample size test here!')
#
#     stoc_factor = 1  # We need to adjust this part later Sep-20
#
#     # G is bound of norm of gradient(\nabla) F_k^i(x) for all k \in [n] and i \in [m] and x \in X
#     G = np.absolute(emp_dist_value).max()  # Also, need to change this funciton add C2
#     # M is an list of bound of |F_k^i(x)| for each i \in [m]
#     absolute_max = np.absolute(emp_dist_value).max(axis=2)
#     sum_over_J = np.sum(absolute_max, axis=1)
#     M = sum_over_J.max(axis=1)
#
#     # Calculate T and our stepsize
#     if ss_type == 'constant':
#         T, R_x, R_p, ss_x, ss_p = R_const_SMD(J, L, n, G, M, delta, rho, obj_tol, C_G, C_K, stoc_factor)
#         print("Max Iteration:", T)
#         print('alg_type:SGD with i_hat')
#         obj_tol = 1e-7
#         ss_x_list = ss_x * np.ones(T + 1)
#         ss_p_list = []
#         for i in range(m):
#             ss_p_list.append(ss_p[i] * np.ones(T + 1))
#
#
#
#
#     elif ss_type == 'diminish':
#         T, R_x, R_p, c_x, c_p = R_dim_SMD(J, L, n, G, M, delta, rho, obj_tol, C_G, C_K,
#                                           stoc_factor)
#         print("Max Iteration:", T)
#         print('alg_type:SGD with i_hat')
#         ss_x_list = c_x * (np.sqrt(1 / (np.arange(T + 1) + 1)))
#         obj_tol = 1e-7
#         ss_p_list = []
#         for i in range(m):
#             ss_p_list.append(c_p[i] * (np.sqrt(1 / (np.arange(T + 1) + 1))))
#
#     print('Rx: %s' % R_x)
#     print('Rp: %s' % R_p)
#
#     # This String List would be used on inf_pi function
#     constr_list = []
#     for i in range(m):
#         constr_list.append('obj_' + str(i))
#
#     bisection = []
#     bisection_count = 0
#
#     vartheta_list = []
#
#     # x_coeff,_ = get_coeff(emp_dist_value_copy,rho,delta, alpha_tol)
#     # print(x_coeff)
#     # Define a PWL to calculate inf_pi
#     list_J = list(range(J))
#     list_L = list(range(L))
#
#     PWL_model = gp.Model('PWL')
#
#     PWL_model.setParam('OutputFlag', 0)
#     var_t = PWL_model.addVar(lb=-GRB.INFINITY, name='t')
#     var_x = PWL_model.addVars(list_J, list_L, ub=1, name='x')
#     PWL_model.addConstrs(gp.quicksum(var_x[j, l] for l in list_L) == 1 for j in list_J)
#
#     # Create dummy constraints
#     for i in range(m):
#         PWL_model.addConstr(var_t >= i, name=constr_list[i])
#
#     obj = var_t
#
#     PWL_model.setObjective(obj, GRB.MINIMIZE)
#     PWL_model.optimize()
#
#     dual_gap_list = []  # each element is a list of dual gap at each bisection iteration
#     iter_timer_list = []  # each element  elapsed time per bisection
#     real_T_list = []  # each element is a list of terminated iteration by dual gap condition at each
#     # bisection iteration
#     early_term_count = 0
#
#     total_tic = time.time()
#
#     sample_freq = 10 ** 4  # We can move this to parameter.
#     sanity_check = 0
#     sanity_freq = 1000
#
#     if dual_gap_option == 2 or dual_gap_option == 0:
#         pass
#     elif dual_gap_option == 3 or dual_gap_option == 1:
#         dual_gap_freq = int(T * dual_gap_freq)
#
#     # Set calculation option
#     if dual_gap_option == 0 or dual_gap_option == 1:
#         dual_gap_cal = 0
#     else:
#         dual_gap_cal = 1
#
#     bisection_count += 1
#     # Change our objective function
#     obj_val = (opt_up + opt_low) / 2
#     RHS[0] = (-1 + 2 * min_flag) * RHS[0]
#     if print_option:
#         print('---------------------------')
#         print('%s-th bisection iteration' % bisection_count)
#         print('alg_type:SGD with i_hat')
#         print('---------------------------')
#         print("current step objective value:", obj_val)
#
#     x = np.empty([dual_gap_freq, J, L])
#     x[0, :, :] = x_0
#     p = np.empty([dual_gap_freq, m, n])
#     p[0, :, :] = p_0
#
#     hat_f_val = np.zeros([dual_gap_freq, m])
#     hat_f_val_ws = np.zeros(m)
#
#     # Variables that is needed to update periodically.
#     iter_count = 0
#     ss_sum_x = 0
#     ss_sum_p = 0  # This does not have to be np.zeros([m])
#     x_ws = np.zeros([J, L])
#     p_ws = np.zeros([m, n])
#
#     # Get x and p for T iterations
#     tic = time.time()
#     time_iter_count = 1
#
#     w_sum = np.zeros(m)
#     w_sum[:] = np.nan
#     w_square_sum = np.zeros(m)
#     w_square_sum[:] = np.nan
#     cum_dist = np.zeros([m, n])
#     cum_dist[:] = np.nan
#
#     for i in range(m):
#         w_sum[i] = np.sum(p_0[i, :])
#         w_square_sum[i] = np.sum(p_0[i, :] ** 2)
#         cum_dist[i, :] = np.cumsum(p_0[i, :])
#
#     dual_gap = []  # List that contains duality gap in this bisection
#
#     sup_val = sup_pi(x_0, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
#     inf_val = inf_pi(p_0, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
#     diff = sup_val - inf_val
#     dual_gap.append(diff)
#     toc = time.time()
#     t = 0
#     while toc - tic < time_cap:
#
#         if t % sample_freq == 0:
#             temp_samples_list = np.random.rand(m, K, sample_freq)
#             grad_samples_list = np.random.rand(K_grad, sample_freq)
#         toc = time.time()
#         if print_option and toc- tic > time_iter_count * time_freq:
#             print('=========================================')
#             print('%s-st iteration start time:' % (t + 1), toc - tic)
#             print('Time Iter Count: %s' %time_iter_count)
#             sup_val = sup_pi(x_ws / ss_sum_x, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
#             inf_val = inf_pi(p_ws / ss_sum_p, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
#             diff = sup_val - inf_val
#             dual_gap.append(diff)
#             print("%s-st iteration duality gap:" % (t + 1), diff)
#             # print("x_bar:", x_bar)
#             # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
#             # print("p_new[0,0,0:]:", p_new[0,0,:])
#             # print("X Coeff:", temp_coeff)
#             print("sup_val:", sup_val)
#             print("inf_val:", inf_val)
#             if K_grad < n:
#                 print("w_sum:", w_sum)
#
#
#
#             time_iter_count +=1
#
#
#         # We do not need information of p_t here. Only cumdist and w_sum is needed.
#         x[(t + 1) % dual_gap_freq, :, :], hat_f_val[t % dual_gap_freq, :], i_hat_flag = SMD_x_K_test(K, K_grad,
#             x[t % dual_gap_freq,:, :], p[t % dual_gap_freq,:, :],ss_x_list[t],temp_samples_list[:, :,t % sample_freq],
#                     grad_samples_list[:,t % sample_freq],RHS,emp_dist_value_copy,w_sum, cum_dist)
#         i_flag_count += i_hat_flag
#
#         for i in range(m):
#             p[(t + 1) % dual_gap_freq, i, :], w_sum[i], w_square_sum[i], cum_dist[i, :] = \
#                 SMD_p(x[t % dual_gap_freq, :, :], p[t % dual_gap_freq, i, :], i, ss_p_list[i][t], delta, rho,
#                       alpha_tol, \
#                       emp_dist_value_copy, w_sum[i], w_square_sum[i], cum_dist[i, :])
#
#         """
#
#         Calculate Duality Gap
#
#         """
#         # Calculate Dual gap
#         if dual_gap_cal and (t + 2) % dual_gap_freq == 0:
#             x_ws, ss_sum_x = bar_calculator_temp(x_ws, x, dual_gap_freq, \
#                                                  ss_x_list[
#                                                  iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
#                                                  ss_sum_x)
#             p_ws, ss_sum_p = bar_calculator_temp(p_ws, p, dual_gap_freq, \
#                                                  ss_p_list[0][
#                                                  iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
#                                                  ss_sum_p)
#             #sup_val = sup_pi(x_ws / ss_sum_x, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
#             #inf_val = inf_pi(p_ws / ss_sum_p, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
#             #diff = sup_val - inf_val
#             #dual_gap.append(diff)
#
#         # if dual_gap_cal and (t + 2) % dual_gap_freq == 0 and print_option:
#         #     print("%s-st iteration duality gap:" % (t + 1), diff)
#         #     # print("x_bar:", x_bar)
#         #     # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
#         #     # print("p_new[0,0,0:]:", p_new[0,0,:])
#         #     # print("X Coeff:", temp_coeff)
#         #     print("sup_val:", sup_val)
#         #     print("inf_val:", inf_val)
#         #     if K_grad < n:
#         #         print("w_sum:", w_sum)
#
#         """
#
#         Update hat_f_val_ws every dual_gap_freq iteration.
#
#         """
#         # Whether dual_gap_cal ==0 or not, we calculate hat_f_val_ws. Also, this update comes later than dual
#         # gap calculation, so we increase our iter_count here.
#         if (t + 1) % dual_gap_freq == 0:
#             hat_f_val_ws += np.tensordot(hat_f_val, ss_x_list[iter_count * dual_gap_freq: \
#                                                               (iter_count + 1) * dual_gap_freq], axes=(0, 0))
#             iter_count += 1
#         t += 1
#
#     dual_gap_list.append(dual_gap)
#     total_toc = time.time()
#     total_solved_time = total_toc - total_tic
#
#     SMD_stat = Statistics(n, m, K, K_grad, ss_type, obj_val, dual_gap_list,
#                           iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_flag_count)
#
#     return SMD_stat



def DRO_SMD_K_test_time(x_0, p_0, emp_dist_value, K, K_grad, delta, rho, alpha_tol, opt_low, opt_up, obj_tol,
        RHS,ss_type, C_K, dual_gap_option, dual_gap_freq, time_cap, time_freq, print_option=1, min_flag=1):
    """
    We implement K_test_time with this function. At every time_freq, we calculate the dual gap and record
    #its value.
    """
    emp_dist_value_copy = emp_dist_value.copy()
    emp_dist_value_copy[0, :, :, :] = (-1 + 2 * min_flag) * emp_dist_value_copy[0, :, :, :]
    RHS[0] = (-1 + 2 * min_flag) * RHS[0]

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
        print("Max Iteration:", T)
        print('alg_type:SGD with i_hat')
        obj_tol = 1e-7
        T_max = 1e7
        ss_x_list = ss_x * np.ones(T_max + 1)
        ss_p_list = []
        for i in range(m):
            ss_p_list.append(ss_p[i] * np.ones(T_max + 1))




    elif ss_type == 'diminish':
        T, R_x, R_p, c_x, c_p = R_dim_SMD(J, L, n, G, M, delta, rho, obj_tol, C_G, C_K,
                                          stoc_factor)
        print("Max Iteration:", T)
        T_max = 1e7
        print('alg_type:SGD with i_hat')
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

    sample_freq = 10 ** 4  # We can move this to parameter.
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

    bisection_count += 1
    # Change our objective function
    obj_val = (opt_up + opt_low) / 2

    if print_option:
        print('---------------------------')
        print('%s-th bisection iteration' % bisection_count)
        print('alg_type:SGD with i_hat')
        print('---------------------------')
        print("current step objective value:", obj_val)

    x = np.empty([dual_gap_freq, J, L])
    x[0, :, :] = x_0
    p = np.empty([dual_gap_freq, m, n])
    p[0, :, :] = p_0

    hat_f_val = np.zeros([dual_gap_freq, m])
    hat_f_val_ws = np.zeros(m)

    # Variables that is needed to update periodically.
    iter_count = 0
    ss_sum_x = 0
    ss_sum_p = 0  # This does not have to be np.zeros([m])
    x_ws = np.zeros([J, L])
    p_ws = np.zeros([m, n])

    # Get x and p for T iterations

    time_iter_count = 1

    w_sum = np.zeros(m)
    w_sum[:] = np.nan
    w_square_sum = np.zeros(m)
    w_square_sum[:] = np.nan
    cum_dist = np.zeros([m, n])
    cum_dist[:] = np.nan

    for i in range(m):
        w_sum[i] = np.sum(p_0[i, :])
        w_square_sum[i] = np.sum(p_0[i, :] ** 2)
        cum_dist[i, :] = np.cumsum(p_0[i, :])

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

    while toc - tic < time_cap:

        if t % sample_freq == 0:
            temp_samples_list = np.random.rand(m, K, sample_freq)
            grad_samples_list = np.random.rand(K_grad, sample_freq)
        toc = time.time()
        if print_option and toc- tic > time_iter_count * time_freq:
            print('=========================================')
            print('%s-st iteration start time:' % (t + 1), toc - tic)
            print('Time Iter Count: %s' %time_iter_count)
            x_ws_list.append(x_ws.copy())
            p_ws_list.append(p_ws.copy())
            ss_sum_x_list.append(ss_sum_x)
            ss_sum_p_list.append(ss_sum_p)
            time_iter_count +=1


        # We do not need information of p_t here. Only cumdist and w_sum is needed.
        x[(t + 1) % dual_gap_freq, :, :], hat_f_val[t % dual_gap_freq, :]= SMD_x(K, K_grad,
            x[t % dual_gap_freq,:, :],ss_x_list[t],temp_samples_list[:, :,t % sample_freq],
                    grad_samples_list[:,t % sample_freq],RHS,emp_dist_value_copy,w_sum, cum_dist)
        for i in range(m):
            p[(t + 1) % dual_gap_freq, i, :], w_sum[i], w_square_sum[i], cum_dist[i, :] = \
                SMD_p(x[t % dual_gap_freq, :, :], p[t % dual_gap_freq, i, :], i, ss_p_list[i][t], delta, rho, RHS,
                      emp_dist_value_copy, w_sum[i], w_square_sum[i], cum_dist[i, :])

        """

        Calculate Duality Gap

        """
        # Calculate Dual gap
        if dual_gap_cal and (t + 2) % dual_gap_freq == 0:
            x_ws, ss_sum_x = bar_calculator_temp(x_ws, x, dual_gap_freq, \
                                                 ss_x_list[
                                                 iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                                 ss_sum_x)
            p_ws, ss_sum_p = bar_calculator_temp(p_ws, p, dual_gap_freq, \
                                                 ss_p_list[0][
                                                 iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                                 ss_sum_p)
            #sup_val = sup_pi(x_ws / ss_sum_x, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
            #inf_val = inf_pi(p_ws / ss_sum_p, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
            #diff = sup_val - inf_val
            #dual_gap.append(diff)

        # if dual_gap_cal and (t + 2) % dual_gap_freq == 0 and print_option:
        #     print("%s-st iteration duality gap:" % (t + 1), diff)
        #     # print("x_bar:", x_bar)
        #     # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
        #     # print("p_new[0,0,0:]:", p_new[0,0,:])
        #     # print("X Coeff:", temp_coeff)
        #     print("sup_val:", sup_val)
        #     print("inf_val:", inf_val)
        #     if K_grad < n:
        #         print("w_sum:", w_sum)

        """

        Update hat_f_val_ws every dual_gap_freq iteration.

        """
        # Whether dual_gap_cal ==0 or not, we calculate hat_f_val_ws. Also, this update comes later than dual
        # gap calculation, so we increase our iter_count here.
        if (t + 1) % dual_gap_freq == 0:
            hat_f_val_ws += np.tensordot(hat_f_val, ss_x_list[iter_count * dual_gap_freq: \
                                                              (iter_count + 1) * dual_gap_freq], axes=(0, 0))
            iter_count += 1
        t += 1

    for idx in range(len(x_ws_list)):
        sup_val = sup_pi(x_ws_list[idx] / ss_sum_x_list[idx], emp_dist_value_copy, rho, delta, alpha_tol, RHS)
        inf_val = inf_pi(p_ws_list[idx] / ss_sum_p_list[idx], emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
        diff = sup_val - inf_val
        dual_gap.append(diff)

    dual_gap_list.append(dual_gap)
    total_toc = time.time()
    total_solved_time = total_toc - total_tic

    SMD_stat = Statistics(n, m, K, K_grad, ss_type, obj_val, dual_gap_list,
                          iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_flag_count)

    return SMD_stat


