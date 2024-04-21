# Stochastic First-Order Algorithms for Constrained Distributionally Robust Optimization

The software and data in this repository are a live version of the software and data
that were used in the research reported on in the paper 
[Stochastic First-Order Algorithms for Constrained Distributionally Robust Optimization](https://doi.org/10.1287/ijoc.2023.0167) by Hyungki Im and Paul Grigas. 

The snapshot of the software and data used in the paper can be found in the [IJOC repository](https://github.com/INFORMSJoC/2023.0167).

## Licensing

### Software License

The source code included in this repository is distributed under the [MIT License](LICENSE).

### Data License

The datasets provided in this repository are made available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE_DATA). Further information is available on the [Creative Commons official page](https://creativecommons.org/licenses/by/4.0/).

## Updates Since Snapshot Version

This section lists the updates made to the repository since the snapshot version was created.

## Description

The goal of this software is to demonstrate the effectiveness of the stochastic first-order method which is proposed in [Stochastic First-Order Algorithms for Constrained Distributionally Robust Optimization]() for constrained distributionally robust optimization.

## Prerequisites

Before you begin, ensure you have the necessary Python packages installed by referring to the `requirements.txt` file. Additionally, licenses for the commercial solvers [Gurobi](https://www.gurobi.com/) and [Mosek](https://www.mosek.com/) are required. Moreover, please unzip the `adult.zip` file under the `data` folder to run the fairness ML example.


## Building

To compile the C code for the RedBlackTree, follow these steps:

1. Create a new virtual environment and install the Python packages specified in `requirements.txt`.
2. Activate the virtual environment and navigate to the `src/RBTree` directory.
3. Execute the `setup.py` script using Python to generate the C code from the `cython_RBTree.pyx` file:

   ```
   python setup.py build_ext --inplace
   ```
## Repository Structure

### Data

The `Data` folder contains the datasets used in our experiments. For the fairness machine learning example, we utilize a customized version of the [adult income dataset](https://archive.ics.uci.edu/dataset/2/adult) from UCI. All other experimental data are generated synthetically during runtime.

### Source Code

The `src` folder houses the source code for each experiment. Each subdirectory contains specific solvers including the stochastic online first-order (SOFO) approach (`ExperimentName_SMD_Solver.py`), the online first-order (OFO) approach (`ExperimentName_FMD_Solver.py`), and various utility functions (`ExperimentName_UBRegret.py`, `ExperimentName_test_functions.py`, `ExperimentName_utils.py`). You can adjust the experiment parameters by modifying `ExperimentName_test_functions.py`.

The `RBTree` subdirectory within `src` implements the RedBlackTree using C code, which must be built as described above.

### Scripts

The `scripts` folder includes Jupyter notebooks for running each experiment. For example, `n_num_test` compares the solving times of SOFO and OFO approaches by varying the number of samples (`n`). The `K_time_test` assesses the duality gap over time with different values of `K`. Detailed explanations are available within each notebook.

## Results

To conduct new experiments, run the corresponding Jupyter Notebook and modify the parameters as necessary. Results will be stored in the `results` folder, and those used in our publication are in the `submitted_results` folder.

## Replication

To replicate the results presented in our paper, adjust the parameters in the notebooks according to the settings specified in `FML_test_functions.py`. For instance, to replicate the results for Figure 2-(a) in the Fairness ML experiment, modify the parameters in `FML_n_num_test.ipynb` as follows:

```
poly_degree = 3
n_list_nt = np.linspace(10000, 45000, 15)
repeats_num = 20
```

<div style="text-align: center;">
    <img src="figures/FML_n_num_test.jpg" alt="Figure 2-(a)" width="700" />
    <p style="text-align: center;">Figure 2-(a): Comparison of solving time between OFO and SOFO</p>
</div>

More details of parameters are presented in each Jupyter notebook. Adjusting these parameters will allow you to closely replicate the experiments and analyze the outcomes as documented in the paper. 

## Support

For support in using this software, submit an
[issue](https://github.com/HyungkiIm/SFOM-DRO/issues).
