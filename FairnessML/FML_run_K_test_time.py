import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from FML_test_functions import *

"""
This code outputs results that is needed to compare the SP gap versus cpu time between SOFO and OFO.
"""

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
data_path = parent_path + '/data/adult/'

#Read Adult Data
poly_degree = 4
df = pd.read_csv(data_path + 'adult_processed_poly={}.csv'.format(poly_degree), index_col=0)
df_np = df.to_numpy()

#Only use the first 3e4 samples

X = df_np[:,:-1]
y = df_np[:,-1]
n_samples = X.shape[0]
n_sst = 45000

seed_sst = 1234
np.random.seed(seed_sst)
#n_list_nt = np.linspace(10000,20000,11)
c = 0.05 #Second and Third Constraint RHS

repeats_sst = 1
K_list_sst = [200, n_sst]
print_opt = 1
warm_start = 0
feas_opt =  1


time_cap_sst = 50
time_freq_sst = 5
dual_gap_freq_list = [250,50]
repeat_number = 0

#Create Emp_dist






data_list = []
#Implement K_test_time
for rep_idx in range(repeats_sst):
    temp_list = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_sst / n_samples, shuffle=True,
                                                        random_state=77)
    for idx, K_sst in enumerate(K_list_sst):
        stat = K_test_time(n_sst, X_train, y_train, c, time_cap_sst, time_freq_sst,dual_gap_freq_list[idx], K_sst, print_opt)
        temp_list = [n_sst, poly_degree, K_sst, time_cap_sst, time_freq_sst]
        temp_list.append(stat.dual_gap_list[0])
        data_list.append(temp_list)

df = pd.DataFrame(data_list, columns=['n', 'deg', 'K', 'time cap', 'time freq', 'Avg_Dual_Gap'])
custom_path = '/results/K_time_result/K_test_time_n={}_poly={}_rdix={}.csv'.format(n_sst, poly_degree,
                                                                                   repeat_number)
save_path = parent_path + custom_path
df.to_csv(save_path, index=False)




