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
