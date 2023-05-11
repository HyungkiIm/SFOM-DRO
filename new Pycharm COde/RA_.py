import numpy as np
import math
#Todo
"""
1. Organize both functions. We need to recheck equations for R_x and R_p both. 
2. Change C2 to CG form. 3
"""


"""

This code include function  R_x_p_combined_constant and  R_x_p_combined_constant


Update Notes:

"""

def R_const_SMD(J, L, n, G, M, delta, rho, obj_tol, C_G, C_K, stoc_factor):


    M_max = max(M)
    D = np.log(L)  # Finite diameter of our primal space
    C2 = C_G / 2
    iter_ub = 1000
    iter_lb = 1

    R_x = J * (C_G * G * math.sqrt(2 * D / iter_ub))
    R_p = 2 * M_max * math.sqrt(rho * C2 / (delta * iter_ub))
    R_total = R_x + R_p - obj_tol * C_K / stoc_factor
    # Double the upperbound until we get a sufficient upperbound
    while R_total >= 0:
        iter_lb = iter_ub
        iter_ub = 2 * iter_ub
        R_x = J * (C_G * G * math.sqrt(2 * D / iter_ub))
        R_p = 2 * M_max * math.sqrt(rho * C2 / (delta * iter_ub))
        R_total = R_x + R_p - obj_tol * C_K / stoc_factor

    # Now perform bisection search

    while iter_ub - iter_lb > 2:
        T = int((iter_ub + iter_lb) / 2)
        R_x = J * (C_G * G * math.sqrt(2 * D / T))
        R_p = 2 * M_max * math.sqrt(rho * C2 / (delta * T))
        R_total = R_x + R_p - obj_tol * C_K / stoc_factor
        if R_total < 0:
            iter_ub = T
        else:
            iter_lb = T

    ss_p = []
    ss_x = math.sqrt(2 * np.log(L) / T) / (C_G * G)
    for i in range(len(M)):
        ss_p.append(1 / (n ** 2 * M[i]) * math.sqrt(rho * delta / (C2 * T)))

    return T, R_x, R_p, ss_x, ss_p

def R_const_FMD(J, L, n, G, M, rho, obj_tol):
    M_max = max(M)
    D = np.log(L)  # Finite diameter of our primal space
    C2 = 2 + math.sqrt(3)
    iter_ub = 1000
    iter_lb = 1

    R_x = J * (G * math.sqrt(2 * D / iter_ub))
    temp = []
    R_p = M_max * math.sqrt(2 * rho / (n * iter_ub))
    R_total = R_x + R_p - obj_tol

    while R_total >= 0:
        iter_lb = iter_ub
        iter_ub = 2 * iter_ub
        R_x = J * (G * math.sqrt(2 * D / iter_ub))
        R_p = M_max * math.sqrt(2 * rho / (n * iter_ub))
        R_total = R_x + R_p - obj_tol

    # Now perform bisection search

    while iter_ub - iter_lb > 2:
        T = int((iter_ub + iter_lb) / 2)
        R_x = J * (G * math.sqrt(2 * D / T))
        R_p = M_max * math.sqrt(2 * rho / (n * T))
        R_total = R_x + R_p - obj_tol
        if R_total < 0:
            iter_ub = T
        else:
            iter_lb = T

    ss_p = []
    ss_x = math.sqrt(2 * np.log(L) / T) / G
    for i in range(len(M)):
        ss_p.append(1 / (M[i] * math.sqrt(n) ** 3) * math.sqrt(2 * rho / T))


    return T, R_x, R_p, ss_x, ss_p

def R_dim_SMD_x(C_g,J,G,D,T): #return R_x and c_x
    return J * C_g * G * math.sqrt(2 * D * (1 + np.log(T))) / (2 * (math.sqrt(T) - 1))

def R_dim_SMD_p(C_g, M, rho, delta, T): #return R_p and c_p
    return M * math.sqrt(rho * C_g * (np.log(T) + 1)) / (math.sqrt(delta) * (math.sqrt(T) - 1))




def R_dim_SMD(J, L, n, G, M, delta, rho, obj_tol, C_g, C_K, stoc_factor):

    M_max = max(M)
    D = np.log(L)  # Finite diameter of our primal space
    C2 = C_g / 2
    iter_ub = 1000
    iter_lb = 1

    R_x = R_dim_SMD_x(C_g,J,G,D,iter_ub)
    R_p = R_dim_SMD_p(C_g, M_max, rho, delta, iter_ub)
    R_total = R_x + R_p - obj_tol * C_K / stoc_factor

    # Double the upperbound until we get a sufficient upperbound\
    while R_total >= 0:
        iter_lb = iter_ub
        iter_ub = 2 * iter_ub
        R_x = R_dim_SMD_x(C_g,J,G,D,iter_ub)
        R_p = R_dim_SMD_p(C_g, M_max, rho, delta, iter_ub)
        R_total = R_x + R_p - obj_tol * C_K / stoc_factor

    # Now perform bisection search

    while iter_ub - iter_lb > 2:
        T = int((iter_ub + iter_lb) / 2)
        R_x = R_dim_SMD_x(C_g,J,G,D,T)
        R_p = R_dim_SMD_p(C_g, M_max, rho, delta, T)
        R_total = R_x + R_p - obj_tol * C_K / stoc_factor
        if R_total < 0:
            iter_ub = T
        else:
            iter_lb = T

    # T = int((iter_ub+iter_lb)/2)
    #Notice that we do not divide c_x by J.
    c_x = math.sqrt(2 * D) / (C_g * G * math.sqrt(np.sum(1 / (np.arange(T) + 1))))
    c_p = []
    for i in range(len(M)):
        c_p.append(1 / (n ** 2 * M[i]) * math.sqrt(delta * rho / (C2 * np.sum(1 / (np.arange(T) + 1)))))

    return T, R_x, R_p, c_x, c_p

def R_dim_FMD(J, L, n, G, M, rho, obj_tol):
    M_max = max(M)
    D = np.log(L)  # Finite diameter of our primal space
    C2 = 2 + math.sqrt(3)
    iter_ub = 1000
    iter_lb = 1

    R_x = J * G * math.sqrt(2 * D * (1 + np.log(iter_ub))) / (2 * (math.sqrt(iter_ub) - 1))
    R_p = M_max * math.sqrt(rho * (np.log(iter_ub) + 1) / (2 * n)) / ((math.sqrt(iter_ub) - 1))
    R_total = R_x + R_p - obj_tol

    while R_total >= 0:
        iter_lb = iter_ub
        iter_ub = 2 * iter_ub
        R_x = J * G * math.sqrt(2 * D * (1 + np.log(iter_ub))) / (2 * (math.sqrt(iter_ub) - 1))
        R_p = M_max * math.sqrt(rho * (np.log(iter_ub) + 1) / (2 * n)) / ((math.sqrt(iter_ub) - 1))
        R_total = R_x + R_p - obj_tol

    # Now perform bisection search

    while iter_ub - iter_lb > 2:
        T = int((iter_ub + iter_lb) / 2)
        R_x = J * G * math.sqrt(2 * D * (1 + np.log(T))) / (2 * (math.sqrt(T) - 1))
        R_p = M_max * math.sqrt(rho * (np.log(T) + 1) / (2 * n)) / ((math.sqrt(T) - 1))
        R_total = R_x + R_p - obj_tol
        if R_total < 0:
            iter_ub = T
        else:
            iter_lb = T

    c_p = []
    c_x = math.sqrt(2 * np.log(L)) / (G * math.sqrt(np.sum(1 / (np.arange(T) + 1))))
    for i in range(len(M)):
        c_p.append(1 / (math.sqrt(n) ** 3 * M[i]) * math.sqrt(2 * rho / (np.sum(1 / (np.arange(T) + 1)))))


    return T, R_x, R_p, c_x, c_p


def R_dim_SMD(J, L, n, G, M, delta, rho, obj_tol, C_g, C_K, stoc_factor):
    M_max = max(M)
    D_x = np.log(L)  # Finite diameter of our primal space
    G_x = C_g * G
    D_p = 4 * rho / (n ** 2)
    G_p = C_g * n * M / delta  # This is np array
    G_p_max = G_p.max()
    iter_ub = 1000
    iter_lb = 1

    R_x = R_dim_SMD_x(J, G_x, D_x, iter_ub)
    R_p = R_dim_SMD_p(G_p_max, D_p, iter_ub)
    R_total = R_x + R_p - obj_tol * C_K / stoc_factor

    # Double the upperbound until we get a sufficient upperbound\
    while R_total >= 0:
        iter_lb = iter_ub
        iter_ub = 2 * iter_ub
        R_x = R_dim_SMD_x(J, G_x, D_x, iter_ub)
        R_p = R_dim_SMD_p(G_p_max, D_p, iter_ub)
        R_total = R_x + R_p - obj_tol * C_K / stoc_factor

    # Now perform bisection search

    while iter_ub - iter_lb > 2:
        T = int((iter_ub + iter_lb) / 2)
        R_x = R_dim_SMD_x(J, G_x, D_x, T)
        R_p = R_dim_SMD_p(G_p_max, D_p, T)
        R_total = R_x + R_p - obj_tol * C_K / stoc_factor
        if R_total < 0:
            iter_ub = T
        else:
            iter_lb = T

    # T = int((iter_ub+iter_lb)/2)
    # Notice that we do not divide c_x by J.
    c_x = math.sqrt(2 * D_x) / (G_x * math.sqrt(np.sum(1 / (np.log(T) + 1))))
    c_p = 1 / G_p * math.sqrt(2 * D_p / (np.log(T) + 1))

    return T, R_x, R_p, c_x, c_p
