import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import gurobipy as gp
from gurobipy import GRB
from statistics import mean

#ToDo


"""

This code includes all the util functions and class that we need.

"""

class Statistics:
    def __init__(self, n, m, K, K_grad, ss_type, last_obj_val, dual_gap_list, \
                 iter_timer_list, total_solved_time, real_T_list,
                 bound_T, R_x, R_p, i_flag_count):
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


# def get_coeff(dataset, rho, delta, alpha_tol):
#     m, J, L, n = dataset.shape
#     opt_p = np.zeros([m, J, L, n])
#     opt_p[:] = np.nan
#     n_list = list(range(n))
#
#     "Notice that our problem is maximization problem"
#     coeff = np.zeros([m, J, L])
#     for i in range(m):
#         for j in range(J):
#             for l in range(L):
#                 lambda0, g_lambda = find_lambda(dataset[i, j, l, :], rho, delta, alpha_tol)
#                 opt_p[i, j, l, :] = np.maximum(dataset[i, j, l, :] / lambda0 + (1 - delta) / n, np.zeros(n)) + delta / n
#
#                 coeff[i, j, l] = np.dot(dataset[i, j, l, :], opt_p[i, j, l, :])
#
#     return coeff, opt_p


# def coeff_calculator(emp_dist_value, p):
#     m, J, L, _ = emp_dist_value.shape
#     coeff_array = np.zeros([m, J, L])
#     for i in range(m):
#         for j in range(J):
#             for l in range(L):
#                 coeff_array[m, j, l] = np.dot(emp_dist_value[i, j, l, :], p[i, j, l, :])
#     return coeff_array


# This function is different from version earlier than 3.7.1
def sup_pi(x_bar, emp_dist_value, rho, delta, alpha_tol, RHS):  # calculate sup_p pi(x_bar,p)
    m, J, L, n = emp_dist_value.shape
    p_coeff = np.zeros([m, n])
    opt_p = np.zeros([m, n])
    val_list = []

    for i in range(m):
        temp = np.tensordot(x_bar, emp_dist_value[i,:,:,:], axes = ([0,1],[0,1])) - RHS[i]
        p_coeff[i, :] = temp

    for i in range(m):
        lambda0, g_lambda = find_lambda(p_coeff[i, :], rho, delta, alpha_tol)
        opt_p[i, :] = np.maximum(p_coeff[i, :] / lambda0 + (1 - delta) / n, np.zeros(n)) + delta / n
        val_list.append(np.dot(opt_p[i, :], p_coeff[i, :]))

    max_val = max(val_list)

    return max_val


def inf_pi(p_bar, dataset, RHS, opt_model, var_t, var_x, constr_list):  # calculates inf_x pi(x,p_bar)
    m, J, L, n = dataset.shape
    coeff = np.zeros([m, J, L])
    for i in range(m):
        coeff[i,:,:] = np.tensordot(p_bar[i,:], dataset[i,:,:,:], axes = (0,2))
    for i in range(m):
        delete_constr = opt_model.getConstrByName(constr_list[i])
        opt_model.remove(delete_constr)
        opt_model.addConstr(var_t >= gp.quicksum(coeff[i, j, l] * var_x[j, l] \
                for j in list(range(J)) for l in list(range(L))) - np.sum(p_bar[i,:]) * RHS[i], name=constr_list[i])
    opt_model.optimize()
    temp = opt_model.objVal
    opt_model.reset(1)
    return temp


def pi_val(x_bar, p_bar, emp_dist_value, RHS):  # Calculates the function pi value

    m, J, L, n = emp_dist_value.shape
    coeff = np.zeros([m, J, L])
    for i in range(m):
        coeff[i,:,:] = np.tensordot(p_bar[i,:], emp_dist_value[i,:,:,:], axes = (0,2))

    max_list = []
    for i in range(m):
        max_list.append(np.sum(coeff[i, :, :] * x_bar) - np.sum(p_bar[i,:]) * RHS[i])

    func_val = max(max_list)

    return func_val

#This function becomes inefficient when t gets large.
# def bar_calculator_list(x, total_iter, ss_list):
#     ss_sum = np.sum(np.array(ss_list[:total_iter]))
#     x_bar = 0
#     for sol_t in range(total_iter):
#         x_bar += ss_list[sol_t] * x[sol_t]
#     x_bar /= ss_sum
#     return x_bar

def bar_calculator(x, total_iter, ss_list):
    x_bar = np.average(x[:total_iter,:,:], axis = 0, weights = ss_list[:total_iter])

    return x_bar

def bar_calculator_temp(prev_x_ws, x, total_iter, ss_list,ss_sum):
    prev_x_ws += np.tensordot(x[:total_iter,:,:], ss_list[:total_iter], axes = (0,0))
    ss_sum += np.sum(ss_list[:total_iter])

    return prev_x_ws, ss_sum