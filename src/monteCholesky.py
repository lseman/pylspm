# Cholesky Pot

# Aguirre-urreta, M. I., & Rönkkö, M. (2015). Sample Size Determination
# and Statistical Power Analysis in PLS Using R : An Annotated Tutorial.
# Communications of the Association for Information Systems, 36(January
# 2015), 33–51. Retrieved from http://aisel.aisnet.org/cais/vol36/iss1/3

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
    cover = []

    for i in range(numRepic):
        data_ = correlated.sample(
            15, replace=False, random_state=(np.random.RandomState()))
        data_.index = range(len(data_))

        bootRes = bootstrap(nrboot, cores, data_, lvmodel,
                            mvmodel, scheme, regression, 0, 100, method='percentile')

        pval.append(bootRes[0])
        cover.append(bootRes[1])

    # Power
    pval = np.array(pval)
    print('Power')
    print(np.sum(pval < 0.05, axis=0) / len(pval))

    # CI 2.5
    ci25 = np.percentile(cover, 2.5, axis=0)
    # CI 97.5
    ci975 = np.percentile(cover, 97.5, axis=0)

    larger = np.sum(cover < ci25, axis=0)
    smaller = np.sum(cover > ci975, axis=0)

    print('Coverage')
    coverage = (len(cover) - larger - smaller) / len(cover)
    print(coverage)
