import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats


def BTS(data):

    n = data.shape[0]
    p = data.shape[1]

    chi2 = -(n - 1 - (2 * p + 5) / 6) * \
        np.log(np.linalg.det(pd.DataFrame.corr(data)))
    df = p * (p - 1) / 2

    pvalue = scipy.stats.distributions.chi2.sf(chi2, df)

    return [chi2, pvalue]


def KMO(data):

    cor_ = pd.DataFrame.corr(data)
    invCor = np.linalg.inv(cor_)
    rows = cor_.shape[0]
    cols = cor_.shape[1]
    A = np.ones((rows, cols))

    for i in range(rows):
        for j in range(i, cols):
            A[i, j] = - (invCor[i, j]) / (np.sqrt(invCor[i, i] * invCor[j, j]))
            A[j, i] = A[i, j]

    num = np.sum(np.sum((cor_)**2)) - np.sum(np.sum(np.diag(cor_**2)))
    den = num + (np.sum(np.sum(A**2)) - np.sum(np.sum(np.diag(A**2))))
    kmo = num / den

    return kmo
