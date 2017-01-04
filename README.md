# PyPLS

**pyplspm** is a Python package dedicated to Partial Least Squares Path Modeling (PLS-PM) analisys using path scheme.

## Dependencies

* SciPy
* Numpy
* Pandas

## Usage

```python
PyPLS(data, LVcsv, MVcsv, max, stopCriterion)
```

Where data, LVcsv and MVcsv are CSV files.

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

# MVcsv file

The MVcsv must contain three collums in the following format:

latent | measurement | mode
------ | ------------| ----
LV1 | indicator1 | A
LV1 | indicator2 | A
LV2 | indicator3 | A
LV3 | indicator4 | A

Defining the indicators connected with each latent variables and the connection mode (A or B).

# References

Library based on Juan Manuel Velasquez Estrada's simplePLS and Gaston Sanchez's plspm made in R
