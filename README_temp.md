# SFOM-DRO
The code for "Stochastic First-order Algorithms for Constrained Distributionally Robust Optimization" by Hyungki Im and Paul Grigas

Dependencies:

Python >= 3.7

CVXPY >= 1.1.17

Mosek >= 10.0.34

NumPy

Pandas 


In the Fairness ML experiment, we utilized the Adult income dataset from UCI [1]. We conducted three experiments, and the code for each of these experiments follows a similar structure. 
As the structure is nearly identical, we provide a brief introduction for each code within the Fairness ML folder.


We compare our approach (SOFO) with its deterministic version (OFO) from [2].

Our codes are largely consisted of three parts: 1) run_test 2) test_functions 3) solver 4) Others
1) run_test: 


FML_run_n_num_test.py: This code outputs results that is needed to compare the solving time between SOFO and OFO for different n:

FML_run_K_test_iter.py: This code outputs results that is needed to compare the convergence rate of SOFO for different K.

FML_run_K_test_time.py: This code outputs results that is needed to compare the SP gap versus cpu time between SOFO and OFO.


2) FML_test_functions.py:


This code includes all the functions that is needed to implement run_test.

3) Solver:


FML_SMD_Solver: Code for SOFO-based Approach.

FML_FMD_Solver: Code for OFO-based Approach.

4) Others:

This includes UBregret.py and utils.py which contains calculation functions for our solvers.




[1] Set, A. D. (2017). UCI machine learning repository. Center for Machine Learning and Intelligent Systems.
 Retrieved from http://archive. ics. uci. edu/ml/datasets/Pima+ Indians+ Diabetes.
 
[2] Ho-Nguyen, Nam, and Fatma Kılınç-Karzan. "Online first-order framework for robust convex optimization." Operations Research 66.6 (2018): 1670-1692.
