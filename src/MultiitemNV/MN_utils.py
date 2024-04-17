import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import gurobipy as gp
from gurobipy import GRB
from statistics import mean

class Statistics:
    def __init__(self, n, m, K, K_grad, ss_type, last_obj_val, dual_gap_list, \
                 iter_timer_list, total_solved_time, real_T_list,
                 bound_T, R_x, R_p, i_equal_count, i_approx_count, i_combined, solved_flag = 0, dual_gap_time=0, p_update_time = 0):
        self.n = n
        self.m = m
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
        self.p_update_time = p_update_time

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

def get_L(c,s,r,b, x, xi):
    if len(xi.shape) > 1:
        n = xi.shape[1]
        y = np.minimum(x.reshape(-1,1), xi)
        Loss_vec = (c-s) @ x + np.tensordot(s-r-b, y, axes = (0,0)) + np.tensordot(b, xi, axes = (0,0))
        return Loss_vec
    else: #xi is 1-dimensional
        y = np.minimum(x, xi)
        Loss = (c - s) @ x + (s - r - b) @ y + b @ xi
        return Loss


#Start from fixing grad_L
def get_grad_L(c,s,r,b, x, xi):
    if len(xi.shape) > 1:
        n = xi.shape[1]
        ones = np.ones(n)
        x_tile = np.outer(x, ones)  # tile x as columns
        max_grad = np.where(x_tile < xi, 1, 0)
        grad_Loss_mat = (c-s).reshape(-1,1) - np.multiply((b+r-s).reshape(-1,1), max_grad)
        return grad_Loss_mat
    else:
        max_grad = np.where(x < xi ,1 , 0)
        grad_Loss = c - s - np.multiply(b+r-s, max_grad)
        return grad_Loss

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
def sup_pi(c_vec, s_vec, r_vec, b_vec, beta, x_bar, emp_dist_value, rho, delta, alpha_tol, RHS):  # calculate sup_p pi(x_bar,p)
    d, n = emp_dist_value.shape
    m = 2
    d += 1
    p_coeff = np.zeros([m, n])
    opt_p = np.zeros([m, n])
    val_list = []

    for i in range(m):

        if i == 0:
            temp = get_L(c_vec, s_vec, r_vec, b_vec,x_bar[1:],emp_dist_value) - RHS[i]
            p_coeff[i,:] = temp
        if i == 1:
            temp = x_bar[0] + 1 / beta * np.maximum(get_L(c_vec, s_vec, r_vec, b_vec,x_bar[1:],emp_dist_value) - x_bar[0],0) - RHS[i]
            p_coeff[i, :] = temp

    for i in range(m):
        lambda0, g_lambda = find_lambda(p_coeff[i, :], rho, delta, alpha_tol)
        opt_p[i, :] = np.maximum(p_coeff[i, :] / lambda0 + (1 - delta) / n, np.zeros(n)) + delta / n
        val_list.append(np.dot(opt_p[i, :], p_coeff[i, :]))
    max_val = max(val_list)

    return max_val


def inf_pi(c_vec, s_vec, r_vec, b_vec, beta, p_bar, dataset, RHS, opt_model, var_x, var_y, var_tau,
           var_z, var_t, constr_list):  # calculates inf_x pi(x,p_bar)
    d, n = dataset.shape
    m = 2
    d += 1
    #Add constraints
    for i in range(m):
        if i == 0:
            # bug fix after the gurobi update to 11.0
            delete_constr = opt_model.getConstrByName(constr_list[i] + "[0]")
            opt_model.remove(delete_constr)
            opt_model.addConstr(var_t >= np.sum(p_bar[i,:]) * (c_vec - s_vec) @ var_x + b_vec.T @ dataset @ p_bar[i,:]\
             - sum((b_vec + r_vec - s_vec)[j] * p_bar[i,:].T @ var_y[j,:] for j in range(d-1)) - np.sum(
                p_bar[i, :]) * RHS[i], name = constr_list[i])

        if i == 1:
            # bug fix after the gurobi update to 11.0
            delete_constr = opt_model.getConstrByName(constr_list[i]+"[0]")
            opt_model.remove(delete_constr)
            opt_model.addConstr(var_t >= np.sum(p_bar[i,:]) * var_tau + 1 / beta * p_bar[i,:] @ var_z - np.sum(
                p_bar[i, :]) * RHS[i], name = constr_list[i])
    opt_model.optimize()
    temp = opt_model.objVal
    opt_model.reset(1)
    return temp


def pi_val(c_vec, s_vec, r_vec, b_vec, beta,x_bar, p_bar, emp_dist_value, RHS):  # Calculates the function pi value

    d, n = emp_dist_value.shape
    d += 1
    m = 2
    max_list = []
    for i in range(m):
        if i == 0:
            temp = p_bar[i,:].T @ get_L(c_vec, s_vec, r_vec, b_vec,x_bar[1:], emp_dist_value) - np.sum(p_bar[i,:]) * RHS[i]
            max_list.append(temp)
        if i == 1:
            temp = p_bar[i,:].T @ (x_bar[0] + 1 / beta * \
            np.maximum(get_L(c_vec, s_vec, r_vec, b_vec,x_bar[1:], emp_dist_value) - x_bar[0],0)) -np.sum(p_bar[i,:]) * RHS[i]
            max_list.append(temp)
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

def dual_gap_calculator(x_ws,x, dual_gap_freq,ss_x_list,ss_sum_x,
                        p_ws,p, ss_p_list, ss_sum_p, multi, addi,
                        c_vec,s_vec,r_vec,b_vec,beta,emp_dist_value_copy,opt_model,var_x,var_y,var_tau,var_z,var_t,constr_list,rho,delta,alpha_tol,RHS):
    x_ws, ss_sum_x = bar_calculator_x(x_ws,x,dual_gap_freq,ss_x_list,ss_sum_x)
    last_p = p[-1,:,:].copy()
    multi_expanded = np.expand_dims(multi, axis=-1)
    addi_expanded = np.expand_dims(addi, axis=-1)
    p *= multi_expanded  # In-place multiplication
    p += addi_expanded  # In-place addition
    actual_p = p
    p_ws, ss_sum_p = bar_calculator_p(p_ws,actual_p,dual_gap_freq,ss_p_list,ss_sum_p)
    sup_val = sup_pi(c_vec, s_vec, r_vec, b_vec, beta,x_ws/ss_sum_x, emp_dist_value_copy, rho, delta, alpha_tol, RHS)
    inf_val = inf_pi(c_vec, s_vec, r_vec, b_vec, beta, p_ws/ss_sum_p, emp_dist_value_copy, RHS,opt_model, var_x, var_y, var_tau,var_z, var_t, constr_list)
    diff = sup_val - inf_val
    p[-1, :, :] = last_p

    return x_ws,ss_sum_x,p_ws,ss_sum_p,sup_val,inf_val,diff