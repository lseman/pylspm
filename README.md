# PyPLS-PM

**pylspm** is a [Python](http://www.python.org/) package dedicated to Partial Least Squares Path Modeling (PLS-PM) analysis.

## Dependencies

* SciPy
* Numpy
* Pandas
* Matplotlib
* Gurobi (for fuzzy regression)

## Usage

```python
# Instantiating the PyPLS class
plspm = PyLSpm(data, LVcsv, MVcsv, scheme, regression, h, maxit, stopCriterion)
```

Where **data**, **LVcsv** and **MVcsv** are CSV files;

**scheme** allows the following options:

* path
* centroid
* factor

**regression** allows the following options:

* ols
* fuzzy

**h** is the h-certain factor used for fuzzy regression only;

**maxit** (default = 300) is the maximum number of iterations allowed until convergence;

and **stopCriterion** (default = 10^-7) is the desired error.

### Data file

Must contain all the indicators with their respective values (*comma-separated values*).

indicator1 | indicator2 | indicator3
-----------|------------|------------
1 | 2 | 1
2 | 3 | 2
1 | 1 | 2

### LVcsv file

The LVcsv must contain two collums in the following format (*comma-separated values*):

source | target
------ | ------
LV1 | LV2
LV1 | LV3

Defining how the latent variables are influencing each other.

### MVcsv file

The MVcsv must contain three collums in the following format (*comma-separated values*):

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
boot = PyLSboot(br, cores, data, LVcsv, MVcsv, scheme, regression, h, maxit, stopCriterion, method, boolen-stine)
```

Where **br** is the number of replications desired;

**cores** is the numbers of cores to use;

**method** allows the following options:

* percentile
* BCa

and **bollen-stine** enables or disables boolen-stine boostraping.

## Additional Methods

Other methods are also available:

Method | Segmentation | Validation | Multi-Group | File
------ | ------------ | ---------- | ----------- | ----
REBUS-PLS | x | | | rebus.py
PLS-GAS | x | | | gac.py
PLS-CPSO | x | | | pso.py
PLS-TABU | x | | | tabu2.py
FIMIX-PLS | x | | | fimix.py
Blindfolding | | x | | blidfolding.py
KMO | | x | | adequacy.py
BTS | | x | | adequacy.py
PCA | | x | | pca.py
Cholesky | | x | | monteCholesky.py
MGA | | | x | mga.py
Permutation | | | x | permuta.py

A missForest (Stekhoven and Bulhmann, 2012) adaptation is available in **imputation.py**.

Multicore is available in bootstraping methods and segmentation methods, also a parallel processing MPI method is implemented in **boot_mpi.py**.

## References

Library inspired by Juan Manuel Velasquez Estrada's simplePLS, Gaston Sanchez's plspm and Mikko Rönkkö's matrixpls made in R.

1. AGUIRRE-URRETA, M. I.; RÖNKKÖ, M. Sample Size Determination and Statistical Power Analysis in PLS Using R : An Annotated Tutorial. Communications of the Association for Information Systems, v. 36, n. January 2015, p. 33–51, 2015. 
2. BREIMAN, L. Random forests. Machine Learning, v. 45, n. 1, p. 5–32, 2001.
3. BROWNLEE, J. Clever Algorithms. In: Search. [s.l: s.n.]. p. 436.
4. CHIN, W. W. How to Write Up and Report PLS Analyses. In: Handbook of # Partial Least Squares. Berlin, Heidelberg: Springer Berlin Heidelberg, # 2010. p. 655–690.
5. HAHN, C. et al. Capturing Customer Heterogeneity Using a Finite Mixture PLS Approach. Schmalenbach Business Review, v. 54, n. July, p. 243–269, # 2002.
6. HENSELER, J.; RINGLE, C. M.; SARSTEDT, M. Testing measurement invariance of composites using partial least squares. International Marketing Review, v. 33, n. 3, p. 405–431, 9 maio 2016.
7. JARBOUI, B. et al. Combinatorial particle swarm optimization (CPSO) for partitional clustering problem. Applied Mathematics and Computation, v. 192, n. 2, p. 337–345, set. 2007.
8. RINGLE, C. M. et al. PLS path modeling and evolutionary segmentation. Journal of Business Research, v. 66, n. 9, p. 1318–1324, set. 2013.
9. SARSTEDT, M. et al. Uncovering and Treating Unobserved Heterogeneity with FIMIX-PLS: Which Model Selection Criterion Provides an Appropriate Number of Segments? Schmalenbach Business Review, v. 63, n. 1, p. 34–62, 2011.
10. SARSTEDT, M.; HENSELER, J.; RINGLE, C. M. Multigroup Analysis in Partial Least Squares (PLS) Path Modeling: Alternative Methods and Empirical Results. In: Measurement and Research Methods in International Marketing (Advances in International Marketing). [s.l: s.n.]. v. 22p. 195–218.
11. STEKHOVEN, D. J.; BUHLMANN, P. MissForest--non-parametric missing value imputation for mixed-type data. Bioinformatics, v. 28, n. 1, p. 112–118, 1 jan. 2012.
12. TRINCHERA, L. Unobserved Heterogeneity in Structural Equation Models: A New Approach to Latent Class Detection in PLS Path Modeling. Tese (Doutorado): Università degli Studi di Napoli Federico II, 2007.