# Cholesky Pot
import pandas
import numpy as np
from numpy import inf
import pandas as pd
import scipy.stats
from scipy.stats import norm

from pylspm import PyLSpm
from boot import PyLSboot
from bootstraping import bootstrap


def monteCholesky(numRepic, nrboot, cores, data_, lvmodel,
                  mvmodel, scheme, regression, h='0', maxit='100'):

    cov = pd.DataFrame.cov(data_)
    chol = np.linalg.cholesky(cov)

    samples = 1000

    uncorrelated = np.random.standard_normal((len(chol), samples))
    correlated = np.dot(chol, uncorrelated)
    correlated = pd.DataFrame(correlated.T, columns=data_.columns)

    pval = []

    for i in range(numRepic):
        data_ = correlated.sample(
            15, replace=False, random_state=(np.random.RandomState()))
        data_.index = range(len(data_))

        pval.append(bootstrap(nrboot, cores, data_, lvmodel,
                              mvmodel, scheme, regression, 0, 100, method='percentile'))

    # Power
    pval = np.array(pval)
    print('Power')
    print(np.sum(pval < 0.05, axis=0) / len(pval))
