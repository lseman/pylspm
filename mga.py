import pandas
import numpy as np
from numpy import inf
import pandas as pd
import scipy.stats
from scipy.stats import norm

from pylspm import PyLSpm
from boot import PyLSboot
from itertools import combinations

def trataGroups(objeto):

    current = list(filter(None.__ne__, objeto))
    current = np.sort(current, axis=0)

    for i in range(len(current[0])):
        current_ = [j[i] for j in current]
        mean_ = np.round(np.mean(current_, axis=0), 4)
        deviation_ = np.round(np.std(current_, axis=0, ddof=1), 4)

    return [mean_, deviation_]

def mga(nrboot, cores, data_, lvmodel,
        mvmodel, scheme, regression, h='0', maxit='100', g1=0, g2=1, segmento='SEM', method='non-parametric'):

    data1 = (data_.loc[data_[segmento] == g1]).drop(segmento, axis=1)
    data2 = (data_.loc[data_[segmento] == g2]).drop(segmento, axis=1)

    estimData1 = PyLSboot(nrboot, cores, data1, lvmodel,
                          mvmodel, scheme, regression, h, maxit)
    estimData2 = PyLSboot(nrboot, cores, data2, lvmodel,
                          mvmodel, scheme, regression, h, maxit)

    estimData1 = estimData1.boot()
    estimData2 = estimData2.boot()

    estimado1 = trataGroups(estimData1)
    estimado2 = trataGroups(estimData2)

    path_diff = np.abs(estimado1[0] - estimado2[0])

    tStat = pd.DataFrame(0, index=range(len(path_diff)),
                         columns=range(len(path_diff)))
    pval = pd.DataFrame(0, index=range(len(path_diff)),
                        columns=range(len(path_diff)))
    df = pd.DataFrame(0, index=range(len(path_diff)),
                      columns=range(len(path_diff)))

    SE1 = estimado1[1]
    SE2 = estimado2[1]

    ng1 = len(data1)
    ng2 = len(data2)
    k1 = ((ng1 - 1) ** 2) / (ng1 + ng2 - 2)
    k2 = ((ng2 - 1) ** 2) / (ng1 + ng2 - 2)
    k3 = np.sqrt(1 / ng1 + 1 / ng2)

    # Parametric

    if method == 'parametric':

        for i in range(len((path_diff))):
            for j in range(len((path_diff))):
                tStat.ix[i, j] = path_diff[i, j] / \
                    (np.sqrt(k1 * SE1[i, j] + k2 * SE2[i, j]) * k3)
                if (tStat.ix[i, j] != 0):
                    pval.ix[i, j] = scipy.stats.t.sf(
                        tStat.ix[i, j], ng1 + ng2 - 2)

    # Non-Parametric

    if method == 'non-parametric':

        for i in range(len((path_diff))):
            for j in range(len((path_diff))):
                tStat.ix[i, j] = path_diff[
                    i, j] / np.sqrt(((ng1 - 1) / ng1) * SE1[i, j] + ((ng2 - 1) / ng2) * SE2[i, j])
                if (tStat.ix[i, j] != 0):
                    df.ix[i, j] = np.round(((((((ng1 - 1) / ng1) * SE1[i, j] + ((ng2 - 1) / ng2) * SE2[i, j])**2) / (
                        ((ng1 - 1) / ng1**2) * SE1[i, j]**2 + ((ng2 - 1) / ng2**2) * SE2[i, j]**2)) - 2), 0)
                    pval.ix[i, j] = scipy.stats.t.sf(
                        tStat.ix[i, j], df.ix[i, j])

#    print('t-stat')
#    print(tStat)
    print('p-value')
    print(pval)

#    print('Paths')
#    print(estimado1[0])
#    print(estimado2[0])
