# PyPLS-PM

**pylspm** is a [Python](http://www.python.org/) package dedicated to Partial Least Squares Path Modeling (PLS-PM) analisys.

## Dependencies

* SciPy
* Numpy
* Pandas
* Matplotlib

## Usage

```python
# Instantiating the PyPLS class
plspm = PyLSpm(data, LVcsv, MVcsv, scheme, regression, maxit, stopCriterion)
```

Where **data**, **LVcsv** and **MVcsv** are CSV files;

**scheme** allows the following options:

* path
* centroid
* factor

**regression** allows the following options:

* ols
* fuzyy

**maxit** (default = 300) is the maximum number of iterations allowed until convergence;

and **stopCriterion** (default = 10^-7) is the desired error.

* The available schemes are: centroid, factor, path and fuzzy.
* The available regressions are OLS and fuzzy.

### Data file

Must contain all the indicators with their respective values.

indicator1 | indicator2 | indicator3
-----------|------------|------------
1 | 2 | 1
2 | 3 | 2
1 | 1 | 2

### LVcsv file

The LVcsv must contain two collums in the following format:

source | target
------ | ------
LV1 | LV2
LV1 | LV3

Defining how the latent variables are influencing each other.

### MVcsv file

The MVcsv must contain three collums in the following format:

latent | measurement | mode
------ | ------------| ----
LV1 | indicator1 | A
LV1 | indicator2 | A
LV2 | indicator3 | A
LV3 | indicator4 | A

Defining the indicators connected with each latent variables and the connection mode (A or B).

## Bootstraping

```python
# Instantiating the PyPLSboot class
boot = PyLSboot(br, cores, data, LVcsv, MVcsv, scheme, regression, maxit, stopCriterion)
```

Where **br** is the number of replications desired;

and **cores** is the numbers of cores to use.

## References

Library based on Juan Manuel Velasquez Estrada's simplePLS, Gaston Sanchez's plspm and Mikko Rönkkö's matrixpls made in R