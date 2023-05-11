import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from statistics import mean
from tqdm import tqdm
from FML_UBRegret import R_dim_SMD
from FML_utils import *
from copy import deepcopy

"""
Fairness ML Experiment
"""




"""
Define Stochastic Mirror Descent function for x
"""

def SMD_x(K, K_grad, x, step_alpha, samples_list, X_train, y_train, s_norm, RHS, w_sum, cum_dist):
    """
    K: per-iteration sample size
    K_grad: batch size of Stochastic FOM
    x: previous x value
    step_alpha: step-size
    samples_list: list of RV of U(0,1) to sample per-iteration batch
    grad_samples_list: list of RV of U(0,1) to sample per-iteration batch
    X_train, y_train, s_norm: data for X, y, normalized sensitive variable
    RHS: RHS of the constraints
    w_sum: w_sum[i]: sum of p^i elements
    cum_dist: cum_dist[i,:]: cumulative distribution of p^i
    """



    # Get the value of m,J,L,n
    n,d = X_train.shape
    m = 3
    F_hat = []

    """
    i_hat calculation
    """
    K_samples = np.zeros([m, K])

    # Pick K samples from each constraint, and j and l.
    for i in range(m):
        K_samples[i, :] = np.digitize(samples_list[i, :], cum_dist[i, :])

    K_samples = np.where(K_samples >= n, n - 1, K_samples).astype(int)

    """
    Compute hat(F)^{i,K} for each constraint i
    hat(F)^{i,K} = 1/K* sum_{j=1}^K F^i(x_t,z^{i,I_j^i})
    """

    for i in range(m):
        if i == 0:
            X_theta = X_train[K_samples[i, :], :] @ x
            f_val = w_sum[i] * np.sum(np.log(1 + np.exp(X_theta)) - np.multiply((1-y_train[K_samples[i, :]]),X_theta))
            f_val = f_val / K - RHS[i] * w_sum[i]
            F_hat.append(f_val)
        elif i == 1:
            X_theta = X_train[K_samples[i, :], :] @ x
            f_val = w_sum[i] * np.sum(np.multiply(s_norm[K_samples[i, :]], X_theta))
            f_val = f_val / K - RHS[i] * w_sum[i]
            F_hat.append(f_val)
        else:
            X_theta = X_train[K_samples[i, :], :] @ x
            f_val = w_sum[i] * np.sum(np.multiply(-s_norm[K_samples[i, :]], X_theta))
            f_val = f_val / K - RHS[i] * w_sum[i]
            F_hat.append(f_val)

    # get the index of max value in list F_hat
    i_hat = F_hat.index(max(F_hat))
    """
    g_t calculation
    """
    K_grad_samples = K_samples[i_hat,:]


    if i_hat == 0:
        X_theta = np.exp(X_train[K_grad_samples,:] @ x)
        g_t = w_sum[i_hat] * X_train[K_grad_samples,:].T @ (X_theta / (1+X_theta) - 1 + y_train[K_grad_samples]) / K_grad
    elif i_hat == 1:
        g_t = w_sum[i_hat] * X_train[K_grad_samples,:].T @ s_norm[K_grad_samples] / K_grad
    else:
        g_t = w_sum[i_hat] * X_train[K_grad_samples,:].T @ -s_norm[K_grad_samples] / K_grad

    """
    x_{t+1} calculation
    """
    x_update = x - step_alpha * g_t

    return x_update, np.array(F_hat)




def SMD_x_K_test(K, K_grad,C_K,obj_tol, x, p, step_alpha, samples_list, grad_samples_list, X_train, y_train, s_norm,
                 RHS, w_sum, cum_dist):
    """
    Refer to SMD_x for more details.
    This function is used to measure the performance of i_hat.
    """

    eps = C_K * obj_tol

    # Get the value of m,J,L,n
    n,d = X_train.shape
    m = 3
    F_hat = []
    F_hat_real = []
    i_hat_flag = 0 # 1 if abs(F_hat_real[i_star] - F_hat[i_hat]) <= eps
    i_hat_equal_flag = 0 # 1 if i_hat == i_star
    i_hat_approx_flag = 0

    """
    i_hat calculation
    """
    K_samples = np.zeros([m, K])

    # Pick K samples from each constraint, and j and l.
    for i in range(m):
        K_samples[i, :] = np.digitize(samples_list[i, :], cum_dist[i, :])

    K_samples = np.where(K_samples >= n, n - 1, K_samples).astype(int)

    """
    Compute hat(F)^{i,K} for each constraint i
    hat(F)^{i,K} = 1/K* sum_{j=1}^K F^i(x_t,z^{i,I_j^i})
    """
    for i in range(m):
        if i == 0:
            X_theta = X_train[K_samples[i, :], :] @ x
            f_val = w_sum[i] * np.sum(np.log(1 + np.exp(X_theta)) - np.multiply((1-y_train[K_samples[i, :]]),X_theta))
            f_val = f_val / K - RHS[i] * w_sum[i]
            F_hat.append(f_val)
        elif i == 1:
            X_theta = X_train[K_samples[i, :], :] @ x
            f_val = w_sum[i] * np.sum(np.multiply(s_norm[K_samples[i, :]], X_theta))
            f_val = f_val / K - RHS[i] * w_sum[i]
            F_hat.append(f_val)
        else:
            X_theta = X_train[K_samples[i, :], :] @ x
            f_val = w_sum[i] * -np.sum(np.multiply(s_norm[K_samples[i, :]], X_theta))
            f_val = f_val / K - RHS[i] * w_sum[i]
            F_hat.append(f_val)
    # get the index of max value in list F_hat
    i_hat = F_hat.index(max(F_hat))

    X_theta = X_train @ x
    for i in range(m):
        if i == 0:
            f_val = p[i,:] @ (np.log(1 + np.exp(X_theta)) - np.multiply(1-y_train,X_theta))
            f_val = f_val - RHS[i] * w_sum[i]
            F_hat_real.append(f_val)
        elif i == 1:
            f_val = p[i, :] @ np.multiply(s_norm, X_theta)
            f_val = f_val - RHS[i] * w_sum[i]
            F_hat_real.append(f_val)
        else:
            f_val = p[i, :] @ -np.multiply(s_norm, X_theta)
            f_val = f_val - RHS[i] * w_sum[i]
            F_hat_real.append(f_val)
    i_star = F_hat_real.index(max(F_hat_real))

    """
    i_hat and i_star comparison
    """

    if abs(F_hat_real[i_star] -F_hat[i_hat]) <= eps:
        i_hat_approx_flag = 1
    if i_star == i_hat:
        i_hat_equal_flag = 1

    """
    g_t calculation
    """
    K_grad_samples = np.zeros(K_grad)
    K_grad_samples[:] = np.digitize(grad_samples_list, cum_dist[i_hat, :])
    K_grad_samples = np.where(K_grad_samples >= n, n - 1, K_grad_samples).astype(int)

    g_t = np.zeros(d)  # subgradient estimator

    if i_hat == 0:
        X_theta = np.exp(X_train[K_grad_samples,:] @ x)
        g_t = w_sum[i_hat] * X_train[K_grad_samples,:].T @ (X_theta / (1+X_theta) - 1 + y_train[K_grad_samples]) / K_grad
    elif i_hat == 1:
        g_t = w_sum[i_hat] * X_train[K_grad_samples,:].T @ s_norm[K_grad_samples] / K_grad
    else:
        g_t = w_sum[i_hat] * X_train[K_grad_samples,:].T @ -s_norm[K_grad_samples] / K_grad

    """
    x_{t+1} calculation
    """
    x_update = x - step_alpha * g_t

    return x_update, np.array(F_hat), i_hat_approx_flag, i_hat_equal_flag


def find_alpha(p_val, w_t, I_t, n, delta, rho, w_sum, w_square_sum):
    """
    This function finds alpha that is needed to update the p_t variables. Please see the Appendix for detailed equations.
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
Stochastic Mirror Descent update for p.
"""
def SMD_p(x, p, i, step_alpha, delta, rho, RHS, X_train, y_train, s_norm, w_sum, w_square_sum, cum_dist):

    n, d = X_train.shape

    #Sample an index
    random_num = np.random.rand()
    I_t = np.digitize(random_num, cum_dist)
    if I_t >=n:
        I_t -=1

    #Calculate the gradient value
    X_theta = np.dot(X_train[I_t, :], x)
    if i == 0:
        grad_val = (np.log(1 + np.exp(X_theta)) - (1-y_train[I_t]) * X_theta-RHS[i]) * w_sum / p[I_t]
    elif i == 1:
        grad_val = (s_norm[I_t] * X_theta - RHS[i]) * w_sum / p[I_t]
    else:
        grad_val = (-s_norm[I_t] * X_theta - RHS[i]) * w_sum / p[I_t]
    old_pval = deepcopy(p[I_t])
    #update p_t to w_t
    p[I_t] += step_alpha * grad_val
    #p[I_t] = delta / n
    # Projection to our chi-square uncertainty set
    # We are not using tree structure here.
    # Note that g'(alpha) is a decreasing function of alpha
    # Input w_sum and w_square_sum are sum of p[t] and square sum of p[t].
    alpha = find_alpha(old_pval, p, I_t, n, delta, rho, w_sum, w_square_sum)

    # Update p value according to alpha that we find above
    # For i != I_t, p_{t+1} = (1-alpha)w_i + alpha/n, Can we reduce this process to O(log n)
    p *= (1 - alpha)
    p += alpha / n
    #Update cumulative distribution and p_sum and p_square_sum
    v1 = alpha * (np.arange(n) + 1) / n
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
    else:  # p_new[I_t] > delta/n
        cum_dist *= (1 - alpha) * w_sum
        temp1 = np.zeros(I_t)
        temp2 = step_alpha * grad_val * np.ones(n - I_t)
        v2 = np.concatenate((temp1, temp2))
        cum_dist += v1 + v2
        w_sum += step_alpha * grad_val
        w_square_sum = w_square_sum - old_pval ** 2 + p[I_t] ** 2 #Recheck this part later.
        w_square_sum = (1 - alpha) ** 2 * w_square_sum + 2 * alpha * (1 - alpha) * w_sum / n + alpha ** 2 / n
        w_sum = (1 - alpha) * w_sum + alpha
        cum_dist /= w_sum
    return p, w_sum, w_square_sum, cum_dist


def DRO_SMD(x_0, p_0, X_train, y_train, K, K_grad, delta, rho, alpha_tol, opt_low, opt_up, obj_tol,
        RHS,ss_type, C_K, dual_gap_option, dual_gap_freq,dual_gap_freq2, print_option=1, K_test_flag=0, min_flag=1,feas_opt = 0,
            warm_start = 0):
    # Change data according to min_flag. // We don't need this for FairnessML example.
    """
    K: per-iteration sample size
    K_grad: batch size for stochastic gradient of x.
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
    # We calculate G and M here
    G = 0.25
    M = np.ones(3) * 0.25
    print('G:', G)
    print('M:', M)

    #Compute T and Stepsize
    if ss_type == 'diminish':
        T, R_x, R_p, c_x, c_p = R_dim_SMD(d, n, G, M, delta, rho, obj_tol, C_G, C_K,
                                          stoc_factor)
        print("Max Iteration:", T)
        print('alg_type:SGD with i_hat')
        ss_x_list = c_x * (np.sqrt(1 / (np.arange(T + 1) + 1))) * 2
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



    sample_freq = 10 ** 4  # We can move this to parameter.
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



    vartheta_flag = 0 #Check Whether hat_vartheta is approximating well.
    feas_flag = 0 #this value must be 0 always. Control feasibility option via parameter feas_opt
    change_flag = 1 #This flag used to change dual_gap_freq to dual_gap_freq2
    # Implementing normal test
    if K_test_flag == 0:

        while opt_up - opt_low > obj_tol and not feas_flag:
            iter_tic = time.time()
            feas_flag = feas_opt
            break_flag = 0
            bisection_count += 1 #first feasibility problem's bisection count is 1.
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

            if warm_start and bisection_count > 1 and change_flag:
                dual_gap_freq = dual_gap_freq2
                change_flag = 0

            #To optimize this code, we save the last dual_gap_freq for x and p
            x = np.empty([dual_gap_freq,d])
            p = np.empty([dual_gap_freq, m, n])
            hat_f_val = np.zeros([dual_gap_freq,m])
            hat_f_val_ws = np.zeros(m)






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
            tic = time.time()

            w_sum = np.zeros(m)
            w_sum[:] = np.nan
            w_square_sum = np.zeros(m)
            w_square_sum[:] = np.nan
            cum_dist = np.zeros([m, n])
            cum_dist[:] = np.nan #Last element should be equal to 1 and the first element should equal to 0.

            for i in range(m):
                w_sum[i] = np.sum(p_0[i, :])
                w_square_sum[i] = np.sum(p_0[i, :] ** 2)
                cum_dist[i, :] = np.cumsum(p_0[i, :]) / w_sum[i]

            dual_gap = []  # List that contains duality gap in this bisection

            # if dual_gap_freq == 0 : #We also need to update this value periodically.
            #
            #     for i in range(m):
            #         f_val[i] += np.sum(np.multiply(np.tensordot(p_old[i, :],emp_dist_value_copy[i, :, :, :], \
            #                                                                  axes=(0, 2)), x_list[0])) * ss_x_list[0]

            for t in range(T):
                # Create samples
                if t % sample_freq == 0:
                    temp_samples_list = np.random.rand(m, K, sample_freq)

                if print_option and (t+2)%dual_gap_freq ==0:
                    toc = time.time()
                    print('='*40)
                    print('%s-st iteration start time:' % (t + 1), toc - tic)

                """
                Update and x and p.
                """
                x[(t+1)%dual_gap_freq,:],hat_f_val[t%dual_gap_freq,:] = SMD_x(K, K_grad, x[t%dual_gap_freq,:],\
                    ss_x_list[t], temp_samples_list[:, :, t % sample_freq], X_train, y_train, s_norm, RHS, w_sum, cum_dist)
                for i in range(m):
                    p[(t+1)%dual_gap_freq,i,:], w_sum[i], w_square_sum[i], cum_dist[i, :] = \
                        SMD_p(x[t%dual_gap_freq,:], p[t%dual_gap_freq,i,:],i,ss_p_list[i][t], delta,rho, RHS, \
                          X_train, y_train, s_norm, w_sum[i], w_square_sum[i], cum_dist[i, :])

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
                    p_ws, ss_sum_p = bar_calculator_p(p_ws, p, dual_gap_freq, \
                                                      ss_p_list[0][
                                                      iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                                      ss_sum_p)
                    dual_gap_toc = time.time()
                    dual_gap_time += dual_gap_toc - dual_gap_tic
                    print('dual_gap_time:',dual_gap_time)

                if dual_gap_cal and (t + 2) % dual_gap_freq == 0:
                    dual_gap_tic = time.time()
                    x_ws, ss_sum_x = bar_calculator_x(x_ws, x, dual_gap_freq,\
                        ss_x_list[iter_count * dual_gap_freq:(iter_count+1) * dual_gap_freq],ss_sum_x)

                    p_ws, ss_sum_p = bar_calculator_p(p_ws,p, dual_gap_freq,\
                                ss_p_list[0][iter_count * dual_gap_freq:(iter_count+1) * dual_gap_freq],ss_sum_p)
                    sup_val = sup_pi(x_ws/ss_sum_x, X_train, y_train, s_norm, rho, delta, alpha_tol, RHS)
                    inf_val = inf_pi(p_ws/ss_sum_p, X_train,y_train,s_norm, RHS)
                    diff = sup_val - inf_val
                    dual_gap.append(diff)
                    dual_gap_toc = time.time()



                    if print_option:
                        print("%s-st iteration duality gap:" % (t + 1), diff)
                        print("Dual Gap Calculation Time %s" % (dual_gap_toc - dual_gap_tic))
                        pi = pi_val(x_ws/ss_sum_x, p_ws/ss_sum_p, X_train, y_train, s_norm, RHS)
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
                # Whether dual_gap_cal ==0 or not, we calculate hat_f_val_ws. Also, this update comes after the  dual
                # gap calculation, so we increase our iter_count here.
                if (t + 1) % dual_gap_freq == 0:
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
                        print("="*40)
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
                        print("="*40)
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

                """"
    
                At the very last iteration. Calculate the duality gap to verify our termination condition.
                Raise error if dual gap > obj_tol
    
                """

                if dual_gap_cal and t == T - 1 and print_option:
                    #array option output
                    real_t = T
                    x_ws, ss_sum_x = bar_calculator_x(x_ws, x, T+1 - iter_count*dual_gap_freq, \
                                    ss_x_list[iter_count * dual_gap_freq:],ss_sum_x)
                    p_ws, ss_sum_p = bar_calculator_p(p_ws, p, T+1- iter_count*dual_gap_freq,\
                                                    ss_p_list[0][iter_count * dual_gap_freq:], ss_sum_p)
                    sup_val = sup_pi(x_ws/ss_sum_x, X_train, y_train, s_norm, rho, delta, alpha_tol, RHS)
                    inf_val = inf_pi(p_ws/ss_sum_p,X_train, y_train, s_norm, RHS)
                    diff = sup_val - inf_val
                    dual_gap.append(diff)


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
                real_t = T
                dual_gap_tic = time.time()
                #Calculate the duality gap at the last iteration if dual_gap_cal == 0
                x_ws, ss_sum_x = bar_calculator_x(x_ws, x, T + 1 - iter_count * dual_gap_freq, \
                                                  ss_x_list[iter_count * dual_gap_freq:], ss_sum_x)
                p_ws, ss_sum_p = bar_calculator_p(p_ws, p, T + 1 - iter_count * dual_gap_freq, \
                                                  ss_p_list[0][iter_count * dual_gap_freq:], ss_sum_p)
                sup_val = sup_pi(x_ws / ss_sum_x, X_train, y_train, s_norm, rho, delta, alpha_tol, RHS)
                inf_val = inf_pi(p_ws / ss_sum_p, X_train, y_train, s_norm, RHS)
                diff = sup_val - inf_val
                dual_gap.append(diff)
                print("%s-st iteration duality gap:" %T, diff)
                dual_gap_toc = time.time()
                dual_gap_time += dual_gap_toc - dual_gap_tic
                print('duality gap computation time:', dual_gap_time)
                if diff < obj_tol/2:
                    solved_flag = 1



            dual_gap_list.append(dual_gap)
            real_T_list.append(real_t)


            if break_flag:
                continue
        total_toc = time.time()
        total_solved_time = total_toc - total_tic - dual_gap_time
        obj_val = (opt_up + opt_low) / 2

        print('Out of %s bisection iteration %s terminated early' % (bisection_count, early_term_count))
        print('Average iteration:', mean(real_T_list))
        print('Total Solved Time:', total_solved_time)
        print('==========================================')

    stat = Statistics(n, m, K, K_grad, ss_type, obj_val, dual_gap_list,
                      iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_flag_count, 0,0,
                      solved_flag=solved_flag, dual_gap_time = dual_gap_time)
    return stat

def DRO_SMD_K_test_iter(x_0, p_0, X_train, y_train, K, K_grad, delta, \
    rho, alpha_tol, opt_low, opt_up, obj_tol, RHS,ss_type, C_K, dual_gap_option, dual_gap_freq, T_cap, print_option=1, min_flag=1):
    """
    Code to compare the convergance rate of the SOFO-based apporach for different values of K.
    """

    n, d = X_train.shape
    s_norm = X_train[:,0] - np.mean(X_train[:,0])
    m = 3

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

    i_approx_count = 0
    i_equal_count = 0
    i_combined_count = 0
    stoc_factor = 1  # 1 is good enough.
    # G is bound of norm of gradient(\nabla) F_k^i(x) for all k \in [n] and i \in [m] and x \in X
    # In Risk-Averse example, we set g to square of 90-th quantile of xi

    # We calculate G and M here
    G = 0.25
    M = np.ones(3) * 0.25
    print('G:', G)
    print('M:', M)





    if ss_type == 'diminish':
        T, R_x, R_p, c_x, c_p = R_dim_SMD(d, n, G, M, delta, rho, obj_tol, C_G, C_K,
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

    ''''''
    RHS[0] = (-1 + 2 * min_flag) * obj_val
    if print_option:
        print('---------------------------')
        print('%s-th bisection iteration' % bisection_count)
        print('alg_type:SGD with i_hat')
        print('---------------------------')
        print("current step objective value:", obj_val)

    # If dual_gap_freq == 0, then we do not save previous output
    x = np.empty([dual_gap_freq, d])
    p = np.empty([dual_gap_freq, m, n])
    hat_f_val = np.zeros([dual_gap_freq, m])
    hat_f_val_ws = np.zeros(m)

    x[0, :] = x_0
    p[0, :, :] = p_0
    # Variables that is needed to update periodically.
    iter_count = 0
    ss_sum_x = 0
    ss_sum_p = 0  # This does not have to be np.zeros([m])
    x_ws = np.zeros(d)
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
    for t in range(T):
        if t % sample_freq == 0:
            temp_samples_list = np.random.rand(m, K, sample_freq)
            grad_samples_list = np.random.rand(K_grad, sample_freq)

        if print_option and (t + 2) % dual_gap_freq == 0:
            toc = time.time()
            print('=========================================')
            print('%s-st iteration start time:' % (t + 1), toc - tic)
        # We do not need information of p_t here. Only cumdist and w_sum is needed.
        x[(t + 1) % dual_gap_freq, :], hat_f_val[t % dual_gap_freq, :], i_hat_approx, i_hat_equal= \
        SMD_x_K_test(K, K_grad, C_K, obj_tol,
                x[t % dual_gap_freq, :], p[t%dual_gap_freq,:,:], ss_x_list[t], temp_samples_list[:, :,t % sample_freq], \
                        grad_samples_list[:, t % sample_freq], X_train, y_train, s_norm, RHS, w_sum, cum_dist)

        for i in range(m):
            p[(t + 1) % dual_gap_freq, i, :], w_sum[i], w_square_sum[i], cum_dist[i, :] = \
                SMD_p(x[t % dual_gap_freq, :], \
                      p[t % dual_gap_freq, i, :], i, ss_p_list[i][t], delta, rho, RHS, \
                      X_train, y_train, s_norm, w_sum[i], w_square_sum[i], cum_dist[i, :])

        i_equal_count += i_hat_equal
        i_approx_count += i_hat_approx
        i_combined_count += max([i_hat_approx,i_hat_equal])


        """

        Calculate Duality Gap

        """
        if dual_gap_cal and (t + 2) % dual_gap_freq == 0:
            dual_gap_tic = time.time()
            x_ws, ss_sum_x = bar_calculator_x(x_ws, x, dual_gap_freq, \
                                              ss_x_list[iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                              ss_sum_x)
            p_ws, ss_sum_p = bar_calculator_p(p_ws, p, dual_gap_freq, \
                                              ss_p_list[0][iter_count * dual_gap_freq:(iter_count + 1) * dual_gap_freq],
                                              ss_sum_p)
            sup_val = sup_pi(x_ws / ss_sum_x, X_train, y_train, s_norm, rho, delta, alpha_tol, RHS)
            inf_val = inf_pi(p_ws / ss_sum_p, X_train, y_train, s_norm, RHS)
            diff = sup_val - inf_val
            dual_gap.append(diff)
            dual_gap_toc = time.time()

            if print_option:
                print("%s-st iteration duality gap:" % (t + 1), diff)
                print("Dual Gap Calculation Time %s" % (dual_gap_toc - dual_gap_tic))
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
                      iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_equal_count, i_approx_count, i_combined_count)

    return SMD_stat

def DRO_SMD_K_test_time(x_0, p_0, X_train,y_train, K, K_grad, delta, rho, alpha_tol, opt_low, opt_up, obj_tol,
        RHS,ss_type, C_K, dual_gap_option, dual_gap_freq, time_cap, time_freq, print_option=1, min_flag=1):
    """
    Function for comparing the convergence speed of SOFO and OFO based approach.
    We implement K_test_time with this function. At every time_freq, we calculate the dual gap and record
    its value.
    """
    n, d = X_train.shape
    s_norm = X_train[:,0] - np.mean(X_train[:,0])
    m = 3
    # Calculate coefficients

    C_G = 1 + math.sqrt(rho / n)

    print('\n')
    print('************************************************************************************')
    print('*******************************Problem Description**********************************')
    print(' ')
    print('Number of constraints: %s, x dimension: %s ,Uncertainty Set Dimension: %s' % (m,d, n))
    print(' ')
    print('*************************************************************************************')
    print('*************************************************************************************')

    i_flag_count = 0
    print('We are doing sample size test here!')

    stoc_factor = 1  # We need to adjust this part later Sep-20

    # We calculate G and M here
    G = 0.25
    M = np.ones(3) * 0.25
    print('G:', G)
    print('M:', M)

    if ss_type == 'diminish':
        T, R_x, R_p, c_x, c_p = R_dim_SMD(d, n, G, M, delta, rho, obj_tol, C_G, C_K,
                                          stoc_factor)
        print("Max Iteration:", T)
        print('alg_type:SGD with i_hat')
        ss_x_list = c_x * (np.sqrt(1 / (np.arange(T + 1) + 1)))
        obj_tol = 1e-7
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
    RHS[0] = (-1 + 2 * min_flag) * obj_val
    if print_option:
        print('---------------------------')
        print('%s-th bisection iteration' % bisection_count)
        print('alg_type:SGD with i_hat')
        print('---------------------------')
        print("current step objective value:", obj_val)

    x = np.empty([dual_gap_freq, d])
    p = np.empty([dual_gap_freq, m, n])
    hat_f_val = np.zeros([dual_gap_freq, m])
    hat_f_val_ws = np.zeros(m)

    # Variables that is needed to update periodically.
    x[0, :] = x_0
    p[0, :, :] = p_0
    # Variables that is needed to update periodically.
    iter_count = 0
    ss_sum_x = 0
    ss_sum_p = 0  # This does not have to be np.zeros([m])
    x_ws = np.zeros(d)
    p_ws = np.zeros([m, n])
    x_list = [x_0]
    p_list = [p_0]

    # Get x and p for T iterations
    tic = time.time()
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

    toc = time.time()
    t = 0
    while toc - tic < time_cap:

        if t % sample_freq == 0:
            temp_samples_list = np.random.rand(m, K, sample_freq)
        toc = time.time()
        if print_option and toc- tic > time_iter_count * time_freq:
            print('=========================================')
            print('%s-st iteration start time:' % (t + 1), toc - tic)
            print('Time Iter Count: %s' %time_iter_count)
            x_list.append(x_ws/ss_sum_x)
            p_list.append(p_ws/ss_sum_p)
            time_iter_count +=1

        """
        Update and x and p.
        """
        x[(t + 1) % dual_gap_freq, :], hat_f_val[t % dual_gap_freq, :] = SMD_x(K, K_grad, x[t % dual_gap_freq, :], \
                                                                               ss_x_list[t],
                                                                               temp_samples_list[:, :, t % sample_freq],
                                                                               X_train, y_train, s_norm, RHS, w_sum,
                                                                               cum_dist)
        for i in range(m):
            p[(t + 1) % dual_gap_freq, i, :], w_sum[i], w_square_sum[i], cum_dist[i, :] = \
                SMD_p(x[t % dual_gap_freq, :], p[t % dual_gap_freq, i, :], i, ss_p_list[i][t], delta, rho, RHS, \
                      X_train, y_train, s_norm, w_sum[i], w_square_sum[i], cum_dist[i, :])

        """

        Calculate Duality Gap

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

        """

        Update hat_f_val_ws every dual_gap_freq iteration.

        """
        if (t + 1) % dual_gap_freq == 0:
            iter_count += 1
        t += 1

    dual_gap_tic = time.time()
    for idx in range(len(x_list)):
        sup_val = sup_pi(x_list[idx], X_train, y_train, s_norm, rho, delta, alpha_tol, RHS)
        inf_val = inf_pi(p_list[idx], X_train, y_train, s_norm, RHS)
        diff = sup_val - inf_val
        if print_option:
            print('sup val:', sup_val)
            print('inf val:', inf_val)
            print('{}-th Dual Gap: {}'.format(idx,diff))
        dual_gap.append(diff)
    dual_gap_toc = time.time()
    dual_gap_time = dual_gap_toc - dual_gap_tic

    dual_gap_list.append(dual_gap)
    total_toc = time.time()
    total_solved_time = total_toc - total_tic

    SMD_stat = Statistics(n, m, K, K_grad, ss_type, obj_val, dual_gap_list,
                          iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_flag_count,0,0,solved_flag = 0, dual_gap_time=dual_gap_time)

    return SMD_stat


