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

## Organization

### Data

The "Data" folder includes the dataset that we used for the experiments. For the fairness machine learning example, we used a customized [adult income dataset](https://archive.ics.uci.edu/dataset/2/adult) from UCI. For the rest of the experiments, data are synthetically generated while running the experiments

### Source

The "src" folder includes the source code for each experiment. The subdirectory for each experiment includes stochastic online first-order (SOFO) approach solver ("SMD_Solver.py"), online first-order (OFO) approach solver ("FMD_Solver.py"), and some util functions ("UBRegret.py", "test_functions.py", "utils.py"). You can customize the parameter settings for each experiment by customizing the helper functions in the "test_functions.py".

Moreover, the "src" folder includes the "RBTree" subdirectory, which implements the RBTree using C code. As written in the "Building" section, you must build the 
"cython_RBTree.pyx" file first to use this. 

## Results

## Replicating

To replicate the results in [Figure 1](results/mult-test), do either

```
make mult-test
```
or
```
python test.py mult
```
To replicate the results in [Figure 2](results/sum-test), do either

```
make sum-test
```
or
```
python test.py sum
```

## Support

For support in using this software, submit an
[issue](https://github.com/tkralphs/JoCTemplate/issues/new).
