
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
from UBRegret import R_x_p_combined_constant, R_x_p_combined_diminish

#Todo
"""
1. Go through Full_alpha function
2. Saving p value in tree structure?
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


"""


class Statistics:
    def __init__(self, n, m, K, K_grad, ss_type, x_bar, p_bar, last_obj_val, x, p, dual_gap_list, \
                 iter_timer_list, total_solved_time, real_T_list,
                 bound_T, R_x, R_p, i_flag_count):
        self.n = n
        self.m = m
        self.K = K
        self.K = K
        self.K_grad = K_grad
        self.ss_type = ss_type
        self.x_bar = x_bar
        self.p_bar = p_bar
        self.last_obj_val = last_obj_val
        self.x = x
        self.p = p
        self.dual_gap_list = dual_gap_list
        self.iter_timer_list = iter_timer_list
        self.total_solved_time = total_solved_time
        self.real_T_list = real_T_list
        self.bound_T = bound_T
        self.R_x = R_x
        self.R_p = R_p
        self.i_flag_count = i_flag_count

    def alg_type(self):
        return_string = ''
        if self.K < self.n and self.K_grad < self.n:
            return_string = 'SGD with i_hat'
        elif self.K == self.n and self.K_grad < self.n:
            return_string = 'SGD with i^*'
        elif self.K < self.n and self.K_grad == self.n:
            return_string = 'FGD with i_hat'
        else:
            return_string = 'FGD with i^*'
        return return_string





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


def SMD_A_x(K, K_grad, x, p, prox, divg, step_alpha, samples_list, RHS, emp_dist_value, w_sum, cum_dist):
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

    if K != n:  # We are using i_hat
        if prox == 'entropy' and divg == 'chi-square':
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
                for j in range(J):
                    for l in range(L):
                        temp += np.sum(emp_dist_value[i, j, l, :][K_samples[i, :].astype(int)]) * x[j, l] * w_sum[i]
                temp = temp / K - RHS[i]
                F_hat.append(temp)

            # get the index of max value in list F_hat
            i_hat = F_hat.index(max(F_hat))

            # Get our subgradient g_t

    #             for j in range(J):
    #                 for l in range(L):
    #                     temp = 0
    #                     temp = np.sum(emp_dist_value[i_hat,j,l,:][K_samples[i,:].astype(int)])*\
    #                     np.sum(p[i_hat,:])
    #                     temp = temp/K
    #                     g_t[j,l] = temp
    else:  # K == n, we don't need random sampling here + we are using i^*
        if prox == 'entropy' and divg == 'chi-square':
            for i in range(m):
                temp = 0
                for j in range(J):
                    for l in range(L):
                        temp += np.dot(emp_dist_value[i, j, l, :], p[i, :]) * x[j, l]
                temp = temp - RHS[i]
                F_hat.append(temp)

            i_hat = F_hat.index(max(F_hat))

    """
    g_t calculation
    """

    # We can also improve this part using pre-sampled indices
    if K_grad != n:  # We are using SGD
        K_grad_samples = np.zeros(K_grad)
        grad_samples_list = np.random.rand(K_grad)
        K_grad_samples[:] = np.digitize(grad_samples_list, cum_dist[i_hat, :])
        K_grad_samples = np.where(K_grad_samples >= n, n - 1, K_grad_samples)
        for k_idx in range(K_grad):
            g_t += w_sum[i_hat] * emp_dist_value[i_hat, :, :, int(K_grad_samples[k_idx])] / K_grad

    else:  # We are using full GD.
        for j in range(J):
            for l in range(L):
                g_t[j, l] = np.dot(emp_dist_value[i_hat, j, l, :], p[i_hat, :])

    """
    x_{t+1} calculation
    """

    # Get next x (Projection Step with entropy prox function)
    theta = np.zeros([J, L])
    for j in range(J):
        for l in range(L):
            theta[j, l] = x[j, l] * np.exp(-step_alpha * g_t[j, l])

    x_update = np.zeros([J, L])
    for j in range(J):
        for l in range(L):
            x_update[j, l] = theta[j, l] / np.sum(theta[j, :])

    return x_update


# In[8]:


# Only used for K_test. Always K,K_grad<n. Return 1 if i_hat == i_*, return 0 otherwise.
def SMD_A_x_K_test(K, K_grad, x, p, prox, divg, step_alpha, RHS, emp_dist_value, w_sum, cum_dist, C_K, obj_tol):
    """
    When Prox function is entropy function
    If we use chi-square as a f-divergence, we need to relax our uncertainty set according to
    Duchi's paper
    """
    # Get the value of m,J,L,n
    m, J, L, n = emp_dist_value.shape
    g_t = np.zeros([J, L])  # subgradient estimator
    F_hat = []
    F_star = []

    """
    i_hat calculation
    """

    K_samples = np.zeros([m, K])
    samples_list = np.random.rand(m, K)

    # Pick K samples from each constraints
    for i in range(m):
        K_samples[i, :] = np.digitize(random_samples[i, :], cum_dist[i, :])

    K_samples = np.where(K_samples >= n, n - 1, K_samples)

    """
    Compute hat(F)^{i,K} for each constraint i
    hat(F)^{i,K} = 1/K* sum_{j=1}^K F^i(x_t,z^{i,I_j^i})
    """

    for i in range(m):
        temp = 0
        for j in range(J):
            for l in range(L):
                temp += np.sum(emp_dist_value[i, j, l, :][K_samples[i, :].astype(int)]) * x[j, l] * w_sum[i]
        temp = temp / K - RHS[i]
        F_hat.append(temp)

    # get the index of max value in list F_hat
    i_hat = F_hat.index(max(F_hat))

    for i in range(m):
        temp = 0
        for j in range(J):
            for l in range(L):
                temp += np.dot(emp_dist_value[i, j, l, :], p[i, :]) * x[j, l]
        temp = temp - RHS[i]
        F_star.append(temp)

    i_star = F_star.index(max(F_star))

    i_flag = 0
    if abs(F_hat[i_hat] - F_star[i_hat]) < (1 - C_K) * obj_tol / 2:
        i_flag = 1

    """
    g_t calculation
    """

    K_grad_samples = np.zeros(K_grad)
    grad_samples_list = np.random.rand(K_grad)
    K_grad_samples[:] = np.digitize(grad_samples_list, cum_dist[i_hat, :])
    K_grad_samples = np.where(K_grad_samples >= n, n - 1, K_grad_samples)

    for k_idx in range(K_grad):
        g_t += w_sum[i_hat] * emp_dist_value[i_hat, :, :, int(K_grad_samples[k_idx])] / K_grad

    """
    x_{t+1} calculation
    """

    # Get next x (Projection Step with entropy prox function)
    theta = np.zeros([J, L])
    for j in range(J):
        for l in range(L):
            theta[j, l] = x[j, l] * np.exp(-step_alpha * g_t[j, l])

    x_update = np.zeros([J, L])
    for j in range(J):
        for l in range(L):
            x_update[j, l] = theta[j, l] / np.sum(theta[j, :])

    return x_update, i_flag


# #First let's just use uniform distribution as x0.
# x_0 = np.ones([J,L])/L
# #x_0 = np.zeros([J,L])
# #x_0[:,0] = 1
# p_0 = np.ones([m,J,L,n])/n
# #Set random seed


# print(x_0)
# #print out x_1
# RHS = np.ones([m])
# RHS[0] = 5
# #SMD_A_x(K, x_0, p_0, 'entropy','chi-square', 0.01, RHS, emp_dist_value)

# Finding alpha such that g'(alpha) = 0
# w is a vector, n is our length of the vector w and delta and rho are given parameters.
# Tol is tolerance of alpha value


# In[9]:


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


def SMD_A_i(x, p, i, divg, step_alpha, delta, rho, alpha_tol, emp_dist_value, w_sum, w_square_sum, cum_dist):
    # for each constraint and JL sample one index I_t ~ p_t^i
    # If we are using chi-square as our f-divergence, we need to relax our uncertainty set.
    _, J, L, n = emp_dist_value.shape

    if divg == 'chi-square':

        #Sample an index
        random_num = np.random.rand()
        I_t = np.digitize(random_num, cum_dist)

        if I_t >=n:
            I_t -=1

        grad_val = np.sum(np.multiply(x, emp_dist_value[i, :, :, I_t])) * w_sum / p[I_t]
        old_pval = p[I_t]
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
            cum_dist += v1
            w_sum = w_sum + step_alpha * grad_val
            cum_dist /= w_sum

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

    return p, w_sum, w_square_sum, cum_dist


# In[11]:


def find_alpha_full(w, n, delta, rho, tol, w_sum_notusing, w_square_sum_notusing):
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


# In[12]:


def Full_A_i(x, p, i, divg, step_alpha, delta, rho, alpha_tol, emp_dist_value, w_sum, w_square_sum, cum_dist):
    # for each constraint and JL sample one index I_t ~ p_t^i
    # If we are using chi-square as our f-divergence, we need to relax our uncertainty set.

    _, J, L, n = emp_dist_value.shape
    if divg == 'chi-square':
        for j in range(J):
            for l in range(L):
                p += step_alpha * x[j, l] * emp_dist_value[i, j, l, :]
        # Projection to our chi-square uncertainty set
        # Here we dont use efficient projection scheme, which uses tree structure

        # Note that g'(alpha) is a decreasing function of alpha
        # Input w_sum and w_square_sum are sum of p[t] and square sum of p[t].

        alpha = find_alpha_full(p, n, delta, rho, alpha_tol, w_sum, w_square_sum)
        # Update p value according to alpha that we find above
        # For i != I_t, p_{t+1} = (1-alpha)w_i + alpha/n
        p *= (1-alpha)
        p += alpha/n

        # For i = I_t, we should first take a look at w_{t,I_t}

        for index in range(n):
            if (1 - alpha) * p[index] + alpha / n < delta / n:
                p[index] = delta / n

    return p, w_sum, w_square_sum, cum_dist


# In[13]:


# SMD_A_i_wotree(x_0, p_0,0,'chi-square', 1, 0.1, 0.5,1e-5,-emp_dist_value).shape





def find_lambda(dataset, rho, delta, alpha_tol):
    n = len(dataset)

    lb = 0
    ub = 500
    lambda0 = (lb + ub) / 2

    g_lambda = 1e3

    while g_lambda > 0:
        u_threshold = ub * (delta - 1) / n
        u_lambda = np.where(dataset >= u_threshold, dataset, 0)
        I_lambda = np.count_nonzero(u_lambda)  # Size of set I(alhpa)
        g_lambda = np.sum(u_lambda ** 2) / (2 * ub ** 2) - rho / n ** 2 + (1 - delta) ** 2 * (n - I_lambda) / (
                2 * n ** 2)
        ub = 2 * ub

    g_lambda = 1e3

    while abs(g_lambda) > alpha_tol and ub - lb > 1:
        lambda0 = (lb + ub) / 2
        u_threshold = lambda0 * (delta - 1) / n
        u_lambda = np.where(dataset >= u_threshold, dataset, 0)
        I_lambda = np.count_nonzero(u_lambda)  # Size of set I(alhpa)
        g_lambda = np.sum(u_lambda ** 2) / (2 * lambda0 ** 2) - rho / n ** 2 + (1 - delta) ** 2 * (n - I_lambda) / (
                2 * n ** 2)
        if g_lambda < 0:
            ub = lambda0
        else:
            lb = lambda0

    return lambda0, g_lambda


def get_coeff(dataset, rho, delta, alpha_tol):
    m, J, L, n = dataset.shape
    opt_p = np.zeros([m, J, L, n])
    opt_p[:] = np.nan
    n_list = list(range(n))

    "Notice that our problem is maximization problem"
    coeff = np.zeros([m, J, L])
    for i in range(m):
        for j in range(J):
            for l in range(L):
                lambda0, g_lambda = find_lambda(dataset[i, j, l, :], rho, delta, alpha_tol)
                opt_p[i, j, l, :] = np.maximum(dataset[i, j, l, :] / lambda0 + (1 - delta) / n, np.zeros(n)) + delta / n

                coeff[i, j, l] = np.dot(dataset[i, j, l, :], opt_p[i, j, l, :])

    return coeff, opt_p


def coeff_calculator(emp_dist_value, p):
    m, J, L, _ = emp_dist_value.shape
    coeff_array = np.zeros([m, J, L])
    for i in range(m):
        for j in range(J):
            for l in range(L):
                coeff_array[m, j, l] = np.dot(emp_dist_value[i, j, l, :], p[i, j, l, :])
    return coeff_array


# This function is different from version earlier than 3.7.1
def sup_pi(x_bar, emp_dist_value, rho, delta, alpha_tol, RHS):  # calculate sup_p pi(x_bar,p)
    m, J, L, n = emp_dist_value.shape
    p_coeff = np.zeros([m, n])
    opt_p = np.zeros([m, n])
    val_list = []

    # TODO
    #Can we make this part more efficient?
    for i in range(m):
        temp = np.zeros(n)
        for j in range(J):
            for l in range(L):
                temp += emp_dist_value[i, j, l, :] * x_bar[j, l]
        p_coeff[i, :] = temp

    for i in range(m):
        lambda0, g_lambda = find_lambda(p_coeff[i, :], rho, delta, alpha_tol)
        opt_p[i, :] = np.maximum(p_coeff[i, :] / lambda0 + (1 - delta) / n, np.zeros(n)) + delta / n
        val_list.append(np.dot(opt_p[i, :], p_coeff[i, :]) - RHS[i])

    max_val = max(val_list)

    return max_val


def inf_pi(p_bar, dataset, RHS, opt_model, var_t, var_x, constr_list):  # calculates inf_x pi(x,p_bar)
    m, J, L, n = dataset.shape
    coeff = np.zeros([m, J, L])
    for i in range(m):
        for j in range(J):
            for l in range(L):
                coeff[i, j, l] = np.dot(dataset[i, j, l, :], p_bar[i, :])
    for i in range(m):
        delete_constr = opt_model.getConstrByName(constr_list[i])
        opt_model.remove(delete_constr)
        opt_model.addConstr(var_t >= gp.quicksum(coeff[i, j, l] * var_x[j, l] \
                for j in list(range(J)) for l in list(range(L))) - RHS[i], name=constr_list[i])
    opt_model.optimize()
    return opt_model.objVal


def pi_val(x_bar, p_bar, emp_dist_value, RHS):  # Calculates the function pi value

    m, J, L, n = emp_dist_value.shape
    coeff = np.zeros([m, J, L])
    for i in range(m):
        for j in range(J):
            for l in range(L):
                coeff[i, j, l] = np.dot(emp_dist_value[i, j, l, :], p_bar[i, :])

    max_list = []
    for i in range(m):
        max_list.append(np.sum(coeff[i, :, :] * x_bar) - RHS[i])

    func_val = max(max_list)

    return func_val

#This function becomes inefficient when t gets large.
def bar_calculator(x, total_iter, ss_list):
    ss_sum = np.sum(np.array(ss_list[:total_iter]))
    x_bar = 0
    for sol_t in range(total_iter):
        x_bar += ss_list[sol_t] * x[sol_t]
    x_bar = x_bar / ss_sum
    return x_bar


# We have to construct our main algorithm using A_x and A_i
# T is total iteration.


def DRO_Solver(x_0, p_0, emp_dist_value, K, K_grad, prox, divg, delta, rho, nu, alpha_tol, opt_low, opt_up, obj_tol,
               RHS,
               ss_type, C_K, dual_gap_freq, T_cap, print_option=1, K_test_flag=0, min_flag=1):


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

    if K_test_flag:
        x_alg = SMD_A_x_K_test
    else:
        x_alg = SMD_A_x

    i_flag_count = 0

    if K_test_flag:
        print('We are doing sample size test here!')

    # Change C_K according to K value
    if K == n:
        C_K = 1

    # Calculate stoc_factor if K_grad < n
    # Currently we just use stoc_factor = 1

    # Omega = np.log(2*m/nu)
    #
    # if K_grad < n:
    #     stoc_factor = 1 + Omega/2 + 4*math.sqrt(Omega)

    stoc_factor = 1  # We need to adjust this part later Sep-20

    if K < n and K_grad < n:
        alg_type = 'SGD with i_hat'
    elif K == n and K_grad < n:
        alg_type = 'SGD with i^*'
    elif K < n and K_grad == n:
        alg_type = 'FGD with i_hat'
    else:
        alg_type = 'FGD with i^*'

    if K_grad < n:
        p_alg = SMD_A_i
    else:
        p_alg = Full_A_i

    # G is bound of norm of gradient(\nabla) F_k^i(x) for all k \in [n] and i \in [m] and x \in X
    G = np.absolute(emp_dist_value).max()  # Also, need to change this funciton add C2
    # M is an list of bound of |F_k^i(x)| for each i \in [m]
    absolute_max = np.absolute(emp_dist_value).max(axis = 2)
    sum_over_J = np.sum(absolute_max, axis = 1)
    M = sum_over_J.max(axis = 1)




    # Calculate T and our stepsize
    if ss_type == 'constant':
        T, R_x, R_p, ss_x, ss_p = R_x_p_combined_constant(J, L, n, G, M, delta, rho, obj_tol, C_G, C_K, stoc_factor,
                                                          K_grad)
        if K_test_flag:
            print("Max Iteration:", T_cap)
            print('alg_type:', alg_type)
            T = T_cap
            obj_tol = 1e-7
            ss_x_list = ss_x * np.ones(T + 1)
            ss_p_list = []
            for i in range(m):
                ss_p_list.append(ss_p[i] * np.ones(T + 1))
        else:
            print("Max Iteration:", T)
            print('alg_type:', alg_type)
            ss_x_list = ss_x * np.ones(T + 1)
            ss_p_list = []
            for i in range(m):
                ss_p_list.append(ss_p[i] * np.ones(T + 1))



    elif ss_type == 'diminish':
        T, R_x, R_p, c_x, c_p = R_x_p_combined_diminish(J, L, n, G, M, delta, rho, obj_tol, C_G, C_K,
                                                        stoc_factor, K_grad)
        if K_test_flag:
            print("Max Iteration:", T_cap)
            print('alg_type:', alg_type)
            T = T_cap
            ss_x_list = c_x * (np.sqrt(1 / (np.arange(T + 1) + 1)))
            obj_tol = 1e-7
            ss_p_list = []
            for i in range(m):
                ss_p_list.append(c_p[i] * (np.sqrt(1 / (np.arange(T + 1) + 1))))

        else:
            print("Max Iteration:", T)
            print('alg_type:', alg_type)
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

    sample_freq = 10**5

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
                print('alg_type:', alg_type)
                print('---------------------------')
                print("current step objective value:", obj_val)

            x = []
            p = []
            x.append(x_0)
            p.append(p_0)
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

                # Now our step-size is not uniform. Make sure to change x_bar and p_bar according to our step size

                #Create samples
                if t% sample_freq == 0:
                    temp_samples_list = np.random.rand(m,K, sample_freq)

                if dual_gap_freq != 0 and t % dual_gap_freq == 0 and print_option:
                    toc = time.time()
                    print('=========================================')
                    print('%s-st iteration start time:' % (t + 1), toc - tic)
                x_new = x_alg(K, K_grad, x[t], p[t], prox, divg, ss_x_list[t],temp_samples_list[:,:,t % sample_freq],\
                              RHS, emp_dist_value_copy, w_sum, cum_dist)
                x.append(x_new)
                tmp = np.zeros([m, n])
                for i in range(m):
                    tmp[i,:], w_sum[i], w_square_sum[i], cum_dist[i,:] = p_alg(x[t], p[t][i,:], i, divg, ss_p_list[i][t],
                                                                                delta,
                                                                                rho, alpha_tol, \
                                                                                emp_dist_value_copy, w_sum[i],
                                                                                w_square_sum[i], cum_dist[i, :])

                    # if abs(np.sum(p_new) - w_sum_temp) > 1e-3:
                    #     print('sum of p:', np.sum(p_new))
                    #     print('w_sum:', w_sum_temp)
                    #     raise TypeError("There is significant differnence")

                p.append(tmp)

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

                if  dual_gap_freq != 0 and t % dual_gap_freq == 0 and print_option:
                    print("%s-st iteration duality gap:" % (t + 1), diff)
                    #print("x_bar:", x_bar)
                    # print("p_bar[0,0,0,:]:",p_bar[0,0,0,:])
                    # print("p_new[0,0,0:]:", p_new[0,0,:])
                    # print("X Coeff:", temp_coeff)
                    print("sup_val:", sup_val)
                    print("inf_val:", inf_val)
                    if K_grad < n:
                        print("w_sum:", w_sum)
                    else:
                        print('w_sum:',np.sum(p[t+1],axis = 1)) #We need to turn this print option later.

                test_freq = 1000
                if print_option and t% test_freq==0:
                    toc = time.time()
                    if K_grad < n:
                        print('=========================================')
                        print('%s-st iteration start time:' % (t + 1), toc - tic)
                        print("w_sum at %s iteration: %s" %(t,w_sum))
                    else:
                        print('=========================================')
                        print('%s-st iteration start time:' % (t + 1), toc - tic)
                        print('w_sum at %s iteration: %s' %(t,np.sum(p[t+1],axis = 1)))
                        #We need to turn this print option later.


                if t == T-1:
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
                    if K_grad < n:
                        print("w_sum:", w_sum)
                    else:
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
            tmp_vartheta_list = []
            theta_list = ss_x_list / np.sum(ss_x_list)

            for i in range(m):
                f_val = 0
                for t in range(T + 1):
                    for j in range(J):
                        for l in range(L):
                            f_val += np.dot(p[t][i, :], emp_dist_value_copy[i, j, l, :]) * x[t][j, l] * theta_list[t]
                f_val = f_val - RHS[i]
                tmp_vartheta_list.append(f_val)

            vartheta_list.append(tmp_vartheta_list)
            vartheta = max(tmp_vartheta_list)

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

    elif K_test_flag == 1:
        bisection_count += 1

        # Change our objective function
        obj_val = (opt_up + opt_low) / 2
        RHS[0] = (-1 + 2 * min_flag) * obj_val
        if print_option:
            print('---------------------------')
            print('%s-th bisection iteration' % bisection_count)
            print('alg_type:', alg_type)
            print('---------------------------')
            print("current step objective value:", obj_val)

        x = []
        p = []
        x.append(x_0)
        p.append(p_0)
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

            # Now our step-size is not uniform. Make sure to change x_bar and p_bar according to our step size

            if t % dual_gap_freq == 0 and print_option:
                toc = time.time()
                print('=========================================')
                print('%s-st iteration start time:' % (t + 1), toc - tic)
            x_new, i_flag = x_alg(K, K_grad, x[t], p[t], prox, divg, ss_x_list[t], RHS, emp_dist_value_copy, \
                                  random_samples_list[t, :, :], w_sum, cum_dist, C_K, obj_tol)
            x.append(x_new)
            i_flag_count += i_flag
            tmp = np.zeros([m, n])
            for i in range(m):
                p_new, w_sum_temp, w_square_sum_temp, cum_dist_temp = p_alg(x[t], p[t], i, divg, ss_p_list[i][t],
                                                                            delta,
                                                                            rho, alpha_tol, \
                                                                            emp_dist_value_copy, w_sum[i],
                                                                            w_square_sum[i], cum_dist[i, :])
                w_sum[i] = w_sum_temp
                w_square_sum[i] = w_square_sum_temp
                cum_dist[i, :] = cum_dist_temp
                # if abs(np.sum(p_new) - w_sum_temp) > 1e-3:
                #     print('sum of p:', np.sum(p_new))
                #     print('w_sum:', w_sum_temp)
                #     raise TypeError("There is significant differnence")
                tmp[i, :] = p_new

            p.append(tmp)

            # Calculate Dual gap
            if t % dual_gap_freq == 0:
                x_bar = bar_calculator(x, t + 2, ss_x_list)
                p_bar = bar_calculator(p, t + 2, ss_p_list[0])
                sup_val = sup_pi(x_bar, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
                inf_val = inf_pi(p_bar, emp_dist_value_copy, RHS, PWL_model, var_t, var_x, constr_list)
                diff = sup_val - inf_val
                dual_gap.append(diff)

        dual_gap_list.append(dual_gap)

        total_toc = time.time()
        total_solved_time = total_toc - total_tic
        obj_val = (opt_up + opt_low) / 2

    stat = Statistics(n, m, K, K_grad, ss_type, x_bar, p_bar, obj_val, x, p, dual_gap_list,
                      iter_timer_list, \
                      total_solved_time, real_T_list, T, R_x, R_p, i_flag_count)

    # Update the last objective value

    # obj_val = (-1 + 2*min_flag) * obj_val

    return stat

"""

Plot 2: different m

"""
#
# # Draw Figure 2
# # stat2_list[m_idx][n_idx][rep_idx][alg_idx]
# for m_num in m_list:
#     stat_m = []
#     for n_num in n_list2:
#         K = 500
#         stat_repeat = []
#
#         # Create our dataset
#         emp_dist_value = np.zeros([m_num, J, L, n_num])
#         emp_dist_value[:] = np.nan
#         p_0 = np.ones([m_num, n_num]) / n_num
#         RHS = np.ones([m_num])
#
#         emp_dist_value_1 = np.random.normal(loc=0.5, scale=0.125, size=(m_num, J, L - 1, n_num))
#         emp_dist_value_2 = np.random.normal(loc=0.25, scale=0.125, size=(m_num, J, 1, n_num))
#         emp_dist_value = np.concatenate([emp_dist_value_1, emp_dist_value_2], axis=2)
#         for rep_idx in range(repeats):
#             stat_temp = []
#
#             for idx in range(alg_num):
#                 alg_stat = DRO_Solver(alg_list[idx][0], alg_list[idx][1], x_0, p_0, emp_dist_value, nu, epsilon,
#                                       obj_tol, \
#                                       K, K_grad, 'entropy', 'chi-square', delta, rho, alpha_tol, opt_low, opt_up,
#                                       obj_tol, \
#                                       RHS, ss_type=ss_type1, min_flag=0)
#
#                 stat_temp.append(alg_stat)
#             stat_repeat.append(stat_temp)
#         stat_m.append(stat_m)
#     stat2_list.append(stat_repeat)
#
# total_runtime_list = []  # total_runtime_list[alg_idx][n_idx][m_idx]
#
# for alg_idx in range(alg_num):
#     alg_time_list = []
#     for n_idx in range(len(n_list2)):
#         n_time_list = []
#         for m_idx in range(len(m_list)):
#             temp = 0
#             for rep_idx in range(repeats):
#                 temp += stat2_list[m_idx][n_idx][rep_idx][alg_idx].total_solved_time
#             temp /= repeats
#             n_time_list.append(temp)
#         alg_time_list.append(n_time_list)
#     total_runtime_list.append(alg_time_list)
#
# # total_runtime_list
#
# plt.figure(figsize=(16, 10))
#
# for n_idx in range(len(n_list2)):
#     plt.subplot(2, math.ceil(len(n_list2) / 2), n_idx + 1)
#     for alg_idx in range(alg_num):
#         plt.plot(m_list, total_runtime_list[alg_idx][n_idx], c=color_list[alg_idx], ls=ls_list[alg_idx], \
#                  marker=marker_list[alg_idx], label=alg_name[alg_idx])
#         plt.xlabel('m')
#         plt.ylabel('time(s)')
#         plt.title('n=%s' % n_list2[n_idx])
#         plt.legend(loc="upper left")
#
# plt.show()


# In[19]:



