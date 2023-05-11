import numpy as np
import cvxpy as cp
import time

#ToDo
"""

1. Change for j in J and l in L using tensordot
2. Change bar_calculator after changing x and p to np array
3. Check if we need get_coeff and coeff_calculator.

"""


"""
This code includes all the util functions and class that we need.
"""

class Statistics:
    def __init__(self, n, m, K, K_grad, ss_type, last_obj_val, dual_gap_list, \
                 iter_timer_list, total_solved_time, real_T_list,
                 bound_T, R_x, R_p, i_equal_count, i_approx_count, i_combined, solved_flag = 0, dual_gap_time=0):
        self.n = n
        self.m = m
        self.K = K
        self.K = K
        self.K_grad = K_grad
        self.ss_type = ss_type
        self.last_obj_val = last_obj_val
        self.dual_gap_list = dual_gap_list
        self.iter_timer_list = iter_timer_list
        self.total_solved_time = total_solved_time
        self.real_T_list = real_T_list
        self.max_iter = bound_T
        self.R_x = R_x
        self.R_p = R_p
        self.i_equal_count = i_equal_count
        self.i_approx_count = i_approx_count
        self.i_combined = i_combined
        self.solved = solved_flag
        self.dual_gap_time = dual_gap_time

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

        if ub > 1e10:
            return ub, g_lambda


    g_lambda = 1e3

    while abs(g_lambda) > alpha_tol and ub - lb > 1:
        lambda0 = (lb + ub) / 2


        if lambda0 > 1e10:
            return lambda0, g_lambda

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

# This function is different from version earlier than 3.7.1
#Sanity checked with cvxpy.

def sup_pi(x_bar, X_train, y_train, s_norm, rho, delta, alpha_tol, RHS):  # calculate sup_p pi(x_bar,p)
    n, d = X_train.shape
    m = 3
    p_coeff = np.zeros([m, n])
    opt_p = np.zeros([m, n])
    val_list = []
    #our_tic = time.time()
    X_theta = X_train @ x_bar
    for i in range(m):
        if i == 0:
            f_0 = np.log(1 + np.exp(X_theta)) - np.multiply(1-y_train,X_theta) - RHS[i]
            p_coeff[i,:] = f_0
        elif i == 1:
            f_1 = np.multiply(s_norm, X_theta) - RHS[i]
            p_coeff[i, :] = f_1
        else:
            f_2 = -np.multiply(s_norm, X_theta) - RHS[i]
            p_coeff[i, :] = f_2


    for i in range(m):
        lambda0, g_lambda = find_lambda(p_coeff[i, :], rho, delta, alpha_tol)
        opt_p[i, :] = np.maximum(p_coeff[i, :] / lambda0 + (1 - delta) / n, np.zeros(n)) + delta / n
        val_list.append(np.dot(opt_p[i, :], p_coeff[i, :]))
    max_val = max(val_list)
    #print('sup_list:', val_list)

    #Sanity Check
    # our_toc = time.time()
    # our_time = our_toc - our_tic
    # #Sanity Check for sup_pi
    # cvx_tic = time.time()
    # cvxpy_list = []
    # for i in range(m):
    #     p = cp.Variable(n)
    #     obj = p @ p_coeff[i,:]
    #     constr = [cp.sum((n*p - 1)**2) <= 2 * rho]
    #     constr += [p >= delta/n]
    #     problem = cp.Problem(cp.Maximize(obj),constr)
    #     problem.solve(solver=cp.MOSEK)
    #     obj_val = problem.value
    #     cvxpy_list.append(obj_val)
    # cvx_toc = time.time()
    # cvxpy_time = cvx_toc - cvx_tic
    # print('our opt val:', val_list)
    # print('our time:', our_time)
    # print('cvxpy opt val:', cvxpy_list)
    # print('cvxpy time:', cvxpy_time)
    # print('pi_val:', np.sum(np.multiply(p_bar,p_coeff),axis = 1))
    return max_val


def inf_pi(p_bar, X_train, y_train, s_norm, RHS):  # calculates inf_x pi(x,p_bar)
    n, d = X_train.shape
    m = 3
    #Define Variables
    theta = cp.Variable(d)
    t = cp.Variable(1)
    X_theta = X_train @ theta
    #Define Problem
    obj = t
    constr = []
    constr += [p_bar[0,:] @ (cp.logistic(X_theta) - cp.multiply(1-y_train, X_theta)-RHS[0]) <= t]
    constr += [p_bar[1,:] @ (cp.multiply(s_norm, X_theta)- RHS[1]) <= t]
    constr += [p_bar[2, :] @ (-cp.multiply(s_norm, X_theta)- RHS[2]) <= t]
    problem = cp.Problem(cp.Minimize(obj), constr)
    problem.solve(solver=cp.MOSEK)
    obj_val = problem.value

    return obj_val


def pi_val(x_bar, p_bar, X_train, y_train, s_norm, RHS):  # Calculates the function pi value

    n, d= X_train.shape
    m = 3
    max_list = []
    X_theta = X_train @ x_bar
    for i in range(m):
        if i == 0:
            f_val = p_bar[i,:].T @ (np.log(1 + np.exp(X_theta)) - np.multiply(1-y_train,X_theta)) - np.sum(p_bar[i,:]) * RHS[i]
            max_list.append(f_val)
        elif i == 1:
            f_val = p_bar[i,:].T @ np.multiply(s_norm,X_theta) -np.sum(p_bar[i,:]) * RHS[i]
            max_list.append(f_val)
        else:
            f_val = p_bar[i, :].T @ -np.multiply(s_norm, X_theta) - np.sum(p_bar[i, :]) * RHS[i]
            max_list.append(f_val)
    print("=====================")
    print(max_list)
    print("======================")
    func_val = max(max_list)

    return func_val

def bar_calculator_x(prev_x_ws, x, total_iter, ss_list,ss_sum):
    prev_x_ws += np.tensordot(x[:total_iter,:], ss_list[:total_iter], axes = (0,0))
    ss_sum += np.sum(ss_list[:total_iter])

    return prev_x_ws, ss_sum

def bar_calculator_p(prev_p_ws, p, total_iter, ss_list,ss_sum):
    prev_p_ws += np.tensordot(p[:total_iter,:], ss_list[:total_iter], axes = (0,0))
    ss_sum += np.sum(ss_list[:total_iter])

    return prev_p_ws, ss_sum