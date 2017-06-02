# Principal Component Analysis
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
from matplotlib import pyplot as plt


def PA(samples, variables):
    datasets = 500
    eig_vals = []

    for i in range(datasets):
        data = np.random.standard_normal((variables, samples))
        cor_ = np.corrcoef(data)
        eig_vals.append(np.sort(np.linalg.eig(cor_)[0])[::-1])

    mean_ = (np.round(np.mean(eig_vals, axis=0), 4))
    return mean_


def PCAdo(block):
    cor_ = np.corrcoef(block.T)
    eig_vals, eig_vecs = np.linalg.eig(cor_)
    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    loadings = (eig_vecs * np.sqrt(eig_vals))

    eig_vals = np.sort(eig_vals)[::-1]
    print('Eigenvalues')
    print(eig_vals)
    print('Variance Explained')
    print(var_exp)
    print('Total Variance Explained')
    print(cum_var_exp)
    print('Loadings')
    print(abs(loadings[:, 0]))

    meanPA = PA(block.shape[0], block.shape[1])

    print('Parallel Analisys')
    pa = (eig_vals - (meanPA - 1))
    print(pa)

    plt.plot(range(len(pa)), pa, '-o')
    plt.plot(np.ones(len(pa)), '--')
    plt.xlabel('Fatores')
    plt.ylabel('Autovalores')
    plt.show()
