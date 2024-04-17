[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Stochastic First-Order Algorithms for Constrained Distributionally Robust Optimization

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[Stochastic First-Order Algorithms for Constrained Distributionally Robust Optimization]() by Hyungki Im and Paul Grigas. 

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

<span style="color: red;">fill here.</span>


<span style="color: red;">fill here.</span>


Below is the BibTex for citing this snapshot of the respoitory.
<span style="color: red;">change this later.</span>

```
@misc{CacheTest,
  author =        {T. Ralphs},
  publisher =     {INFORMS Journal on Computing},
  title =         {{CacheTest}},
  year =          {2020},
  doi =           {10.1287/ijoc.2019.0000.cd},
  url =           {https://github.com/INFORMSJoC/2019.0000},
  note =          {Available for download at https://github.com/INFORMSJoC/2019.0000},
}  
```

## Description

The goal of this software is to demonstrate the effectiveness of the stochastic first-order method which is proposed in [Stochastic First-Order Algorithms for Constrained Distributionally Robust Optimization]() for constrained distributionally robust optimization.
## Building

Please see requirements.txt and install the required Python packages. You also need the license for the commercial solver [Gurobi](https://www.gurobi.com/) and [Mosek](https://www.mosek.com/).
Moreover, please follow the following steps to build C code for RedBlackTree. 

1. Create a new virtual environment and download the required Python packages in requirements.txt

2. Activate the newly created virtual environment and navigate to the 'src/RBTree' directory.

3. Run the 'setup.py' script with Python. This will generate the C code from the cython_RBTree.pyx file. Use the following command:

```
python setup.py build_ext --inplace
```

## Contents

### Data

The "Data" folder includes the dataset that we used for the experiments. For the fairness machine learning example, we used a customized [adult income dataset](https://archive.ics.uci.edu/dataset/2/adult) from UCI. For the rest of the experiments, data are synthetically generated while running the experiments

### Source

The "src" folder includes the source code for each experiment. The subdirectory for each experiment includes stochastic online first-order (SOFO) approach solver ("SMD_Solver.py"), online first-order (OFO) approach solver ("FMD_Solver.py"), and some util functions ("UBRegret.py", "test_functions.py", "utils.py"). You can customize the parameter settings for each experiment by customizing the helper functions in the "test_functions.py".

Moreover, the "src" folder includes the "RBTree" subdirectory, which implements the RBTree using C code. As written in the "Building" section, you must build the 
"cython_RBTree.pyx" file first to use this. 

### Scripts

The "scripts" folder includes the Jupyter notebooks to run each experiment. The experiment named "n_num_test" runs the SOFO approach and OFO approach for different values of the number of samples ($n$) to compare its solving time. The experiment named "K_time_test" runs the SOFO approach with different values of $K$ and compares its duality gap over time. More detailed explanations are available in each notebook.

## Results

You can run new experiments by running the corresponding Jupyter Notebook. We specify the parameters that you might want to change for each experiment in the Jupyter Notebook. The newly created results will be saved under the "results" folder. Also, the results that we used in our paper are stored under the "submitted_results" folder.

## Replicating

The parameter set by test_functions.py follows the same parameter setting that we have used in our experiment. Therefore, we only need to change the parameters in the Notebooks to replicate the results.

![Figure 2-(a)](results/mult-test.png)

For example, to replicate the experiment results used for Figure 2-(a) in the Fairness ML experiment, set the parameters in the "FML_n_num_test.ipynb" as follows: 
```
n_list_nt = np.linspace(10000,45000,15)
repeats_num = 20
```

Similar things apply to different experiments.
## Support

For support in using this software, submit an
[issue]().
