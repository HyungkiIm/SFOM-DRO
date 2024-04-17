from cython_RBTree import RedBlackTree
from PS_UBRegret import R_dim_SMD, R_const_SMD
from PS_utils import Statistics, dual_gap_calculator, pi_val
import numpy as np
import math
import time 
import gurobipy as gp
from gurobipy import GRB


# Here K_samples and K_grad_samples are list
def SMD_x_Tree(K, K_grad, x, step_alpha,p_tree_list, coin_K_arr, RHS, emp_dist_value, w_sum):
    """
    Stochastic Mirror Descent for x using RBTree

    Args:
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
    m, J, L, n = emp_dist_value.shape
    F_hat = []

    #get K_samples
    K_samples = []
    for i in range(m):
        K_samples.append(p_tree_list[i].random_sample(coin_K_arr[i, :, :]))

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
        temp = temp / K - RHS[i] * w_sum[i]
        F_hat.append(temp)
    # get the index of max value in list F_hat
    i_hat = F_hat.index(max(F_hat))

    """
    g_t calculation
    """
    g_t = w_sum[i_hat] * np.sum(emp_dist_value[i_hat,:,:,K_samples[i_hat][:K_grad]],axis=0) / K_grad


    """
    x_{t+1} calculation
    """

    # Get next x (Projection Step with entropy prox function)
    theta = np.multiply(x, np.exp(-step_alpha * g_t))
    x_update = np.zeros([J, L])
    for j in range(J):
        x_update[j, :] = theta[j, :] / np.sum(theta[j, :])

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
# In[10]:

# Now the input p is a tree
def SMD_p_Tree(x, p_tree, p_arr, step_alpha, delta, rho, RHS, emp_dist_value, w_sum, w_square_sum):
    # for each constraint and JL sample one index I_t ~ p_t^i
    # If we are using chi-square as our f-divergence, we need to relax our uncertainty set.
    J, L, n = emp_dist_value.shape

    coin_list = np.random.uniform(size=math.ceil(2 * np.log2(n + 1))).reshape((1, math.ceil(2 * np.log2(n + 1))))

    I_t = p_tree.random_sample(coin_list)[0]
    multi = p_tree.multi
    addi = p_tree.addi

    old_pval = multi * p_arr[I_t] + addi

    grad_val = (np.sum(np.multiply(x, emp_dist_value[:, :, I_t]))-RHS) * w_sum / old_pval
    # update p_t to w_t
    w_temp = old_pval + step_alpha * grad_val
    # Projection to our chi-square uncertainty set
    # Note that g'(alpha) is a decreasing function of alpha
    # Input w_sum and w_square_sum are sum of p[t] and square sum of p[t].
    alpha = find_alpha(old_pval, w_temp, n, delta, rho, w_sum, w_square_sum)
    adjusted_w = w_temp * (1-alpha) + alpha/n
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

    else:  # p_new[I_t] > delta/n // We do not need to update tree. Updating multi and addi is sufficient.
        # #update p_arr[I_t] according to the updated multi and addi factor
        p_tree.delete_node(p_arr[I_t], I_t)
        p_tree.insert_node((adjusted_w - p_tree.addi) / p_tree.multi, I_t)
        p_arr[I_t] = (adjusted_w - p_tree.addi) / p_tree.multi
        w_sum = w_sum + step_alpha * grad_val
        w_square_sum = w_square_sum - old_pval ** 2 + adjusted_w ** 2  # Recheck this part later.
        w_square_sum = (1 - alpha) ** 2 * w_square_sum + 2 * alpha * (1 - alpha) * w_sum / n + alpha ** 2 / n
        w_sum = (1 - alpha) * w_sum + alpha

    return p_arr, w_sum, w_square_sum


def DRO_SMD(x_0, p_0, emp_dist_value, K, K_grad, delta, rho, alpha_tol, opt_low, opt_up, obj_tol,
                 RHS, ss_type, C_K, dual_gap_option, dual_gap_freq, T_cap, print_option=1, K_test_flag=0, min_flag=1,
                 feas_opt=0):
    """
    Args:
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
        T_cap: maximum iteration
        print_option: print option
        K_test_flag: flag for K_test
        min_flag: flag for minimization
        feas_opt: feasibility option
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
        T = int(T/4)
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

    sample_freq = 10 ** 5  # We can move this to parameter.

    if dual_gap_option == 2 or dual_gap_option == 0:
        pass
    elif dual_gap_option == 3 or dual_gap_option == 1:
        dual_gap_freq = int(T * dual_gap_freq)

    # Set calculation option
    if dual_gap_option == 0 or dual_gap_option == 1:
        dual_gap_cal = 0
    else:
        dual_gap_cal = 1

    #Check Whether hat_vartheta is approximating well.
    vartheta_flag = 0
    feas_flag = 0

    while opt_up - opt_low > obj_tol and not feas_flag:
        feas_flag = feas_opt
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

        # If dual_gap_freq == 0, then we do not save previous output
        # Todo
        # We are saving x_t only. Exclude this step when it is needed.

        x = np.empty([dual_gap_freq,J, L])
        x[0,:,:] = x_0
        # p keeps tracks of base_p value
        p = np.empty([dual_gap_freq, m, n])
        # multi keep tracks of multi values for each tree
        multi = np.empty([dual_gap_freq,m])
        multi[0,:] = 1
        # addi keep tracks of addi values for each tree
        addi = np.empty([dual_gap_freq,m])
        addi[0,:] = 0
        p[0,:,:] = p_0

        hat_f_val = np.zeros([dual_gap_freq, m])
        hat_f_val_ws = np.zeros(m)

        #Initialize the Tree
        p_tree_list = []
        # bst.construct intial weight equal to p_0
        for i in range(m):
            bst = RedBlackTree()
            bst.construct_tree_py(n)
            p_tree_list.append(bst)

        #Variables that is needed to update periodically.
        iter_count = 0
        ss_sum_x = 0
        ss_sum_p = 0 #This does not have to be np.zeros([m])
        x_ws = np.zeros([J,L])
        p_ws = np.zeros([m,n])

        # Get x and p for T iterations
        coin_K_arr = np.random.uniform(size=[sample_freq, m, K, math.ceil(2 * np.log2(n + 1))])
        tic = time.time()

        w_sum = np.zeros(m)
        w_sum[:] = np.nan
        w_square_sum = np.zeros(m)
        w_square_sum[:] = np.nan

        for i in range(m):
            w_sum[i] = np.sum(p_0[i, :])
            w_square_sum[i] = np.sum(p_0[i, :] ** 2)

        dual_gap = []  # List that contains duality gap in this bisection

        total_dual_gap_time = 0
        for t in range(T):
            if print_option and (t+2)%dual_gap_freq ==0:
                toc = time.time()
                print('=========================================')
                print('%s-st iteration start time:' % (t + 1), toc - tic)
                print('%s-st iteration start time excluding dual_gap_time:' % (t + 1), toc - tic - total_dual_gap_time)

            # Update x
            x[(t+1)%dual_gap_freq,:,:], hat_f_val[t%dual_gap_freq,:] = SMD_x_Tree(K, K_grad, x[t%dual_gap_freq,:,:],\
                                            ss_x_list[t],p_tree_list, coin_K_arr[t%sample_freq,:,:,:], RHS, emp_dist_value_copy, w_sum)

            # Update p
            for i in range(m):
                p[(t+1)%dual_gap_freq,i,:], w_sum[i], w_square_sum[i] = SMD_p_Tree(x[t%dual_gap_freq,:,:], p_tree_list[i], p[t%dual_gap_freq,i,:], \
                                                        ss_p_list[i][t], delta, rho, RHS[i], emp_dist_value_copy[i, :, :, :],
                                                        w_sum[i], w_square_sum[i])
                #update addi and multi
                multi[(t+1)%dual_gap_freq,i] = p_tree_list[i].multi
                addi[(t + 1) % dual_gap_freq, i] = p_tree_list[i].addi

            #Calculate duality gap

            if dual_gap_cal and (t + 2) % dual_gap_freq == 0:
                dual_gap_tic = time.time()
                x_ws, ss_sum_x, p_ws, ss_sum_p, sup_val,inf_val, diff = dual_gap_calculator(
                    x_ws, x, dual_gap_freq, ss_x_list[iter_count * dual_gap_freq:(iter_count+1) * dual_gap_freq], ss_sum_x,
                    p_ws, p, ss_p_list[0][iter_count * dual_gap_freq:(iter_count+1) * dual_gap_freq], ss_sum_p, multi, addi,
                    emp_dist_value_copy, rho, delta, alpha_tol, RHS, PWL_model, var_t, var_x, constr_list
                )
                dual_gap.append(diff)
                dual_gap_toc = time.time()
                total_dual_gap_time += (dual_gap_toc - dual_gap_tic)

            if dual_gap_cal and (t+2) % dual_gap_freq == 0 and print_option:
                print("%s-st iteration duality gap:" % (t + 1), diff)
                print("Dual Gap Calculation Time %s" %(dual_gap_toc - dual_gap_tic))
                print("Total Dual Gap Calculation Time %s" %total_dual_gap_time)
                pi = pi_val(x_ws/ss_sum_x, p_ws/ss_sum_p, emp_dist_value_copy, RHS)
                print("sup_val:", sup_val)
                print("inf_val:", inf_val)
                print('pi_val:', pi)

            """

            Update hat_f_val_ws every dual_gap_freq iteration.

            """
            # Whether dual_gap_cal ==0 or not, we calculate hat_f_val_ws. Also, this update comes later than dual
            # gap calculation, so we increase our iter_count here.
            if (t + 1) % dual_gap_freq == 0:
                hat_f_val_ws += np.tensordot(hat_f_val, ss_x_list[iter_count * dual_gap_freq: \
                                                                    (iter_count + 1) * dual_gap_freq], axes=(0, 0))
                iter_count += 1

            # Check Termination Condition
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
                dual_gap_tic = time.time()
                x_ws, ss_sum_x, p_ws, ss_sum_p, sup_val,inf_val, diff = dual_gap_calculator(
                    x_ws, x, dual_gap_freq, ss_x_list[iter_count * dual_gap_freq:(iter_count+1) * dual_gap_freq], ss_sum_x,
                    p_ws, p, ss_p_list[0][iter_count * dual_gap_freq:(iter_count+1) * dual_gap_freq], ss_sum_p, multi, addi,
                    emp_dist_value_copy, rho, delta, alpha_tol, RHS, PWL_model, var_t, var_x, constr_list
                )
                dual_gap.append(diff)
                dual_gap_toc = time.time()
                total_dual_gap_time += (dual_gap_toc - dual_gap_tic)


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

        dual_gap_list.append(dual_gap)
        real_T_list.append(real_t)

        if break_flag:
            continue

        #Calculate the last hat_f_val_ws

        hat_f_val_ws += np.tensordot(hat_f_val[: T % dual_gap_freq], ss_x_list[iter_count * dual_gap_freq:T], axes=(0, 0))
        ss_sum_x = np.sum(ss_x_list[:T])
        hat_f_val_ws /= ss_sum_x

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

    print('==========================================')

    stat = Statistics(n, m, K, K_grad, ss_type, obj_val, dual_gap_list,
                      iter_timer_list, total_solved_time, real_T_list, T, R_x, R_p, i_flag_count,
                      dual_gap_time= total_dual_gap_time,
                      )
    return stat
