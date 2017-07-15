import pandas
import numpy as np
from numpy import inf
import pandas as pd
import scipy.stats
from scipy.stats import norm

from pylspm import PyLSpm
from boot import PyLSboot

def bootstrap(nrboot, cores, data_, lvmodel,
              mvmodel, scheme, regression, h='0', maxit='100', method='percentile', boolen_stine=0):

    if boolen_stine == 1:
        segmento = 'SEM'
        data_boolen = data_.drop(segmento, axis=1)

        colunas = data_boolen.columns
        S = pd.DataFrame.cov(data_boolen)
        chol = np.linalg.cholesky(S)
        A = (pd.DataFrame(np.linalg.inv(chol)))

        boolen = PyLSpm(data_boolen, lvmodel, mvmodel, scheme, regression, 0, 100)
        implied = np.sqrt(boolen.implied())

        data_boolen = data_boolen - data_boolen.mean()
        data_boolen = np.dot(np.dot(data_boolen, A), implied)

        data_ = pd.DataFrame(data_boolen, columns=colunas)

    tese = PyLSboot(nrboot, cores, data_, lvmodel,
                    mvmodel, scheme, regression, 0, 100)
    resultados = tese.boot()

    default = PyLSpm(data_, lvmodel, mvmodel, scheme,
                     regression, h, maxit).path_matrix.values

    current = list(filter(None.__ne__, resultados))
    current = np.sort(current, axis=0)

    if (method == 'percentile'):
        for i in range(len(current[0])):
            current_ = [j[i] for j in current]
            print('MEAN')
            print(np.round(np.mean(current_, axis=0), 4))
            print('STD')
            print(np.round(np.std(current_, axis=0, ddof=1), 4))
            print('CI 2.5')
            print(np.round(np.percentile(current_, 2.5, axis=0), 4))
            print('CI 97.5')
            print(np.round(np.percentile(current_, 97.5, axis=0), 4))

            print('t-value')
            tstat = np.nan_to_num(np.mean(current_, axis=0) /
                                  np.std(current_, axis=0, ddof=1))
            print(tstat)

            print('p-value')
            pvalue = np.round((scipy.stats.t.sf(
                tstat, len(current_) - 1)), 5)
            print(pvalue)

            return pvalue

    elif (method == 'bca'):

        default = PyLSpm(data_, lvmodel, mvmodel, scheme,
                         regression, 0, 100).path_matrix.values
        for i in range(len(current[0])):
            current_ = [j[i] for j in current]

            alpha = 0.05
            if np.iterable(alpha):
                alphas = np.array(alpha)
            else:
                alphas = np.array([alpha / 2, 1 - alpha / 2])

            # bias
            z0 = norm.ppf(
                (np.sum(current_ < default, axis=0)) / len(current_))
            zs = z0 + \
                norm.ppf(alphas).reshape(alphas.shape + (1,) * z0.ndim)

            # acceleration and jackknife
            jstat = PyLSboot(len(data_), cores, data_,
                             lvmodel, mvmodel, scheme, regression, 0, 100)
            jstat = jstat.jk()
            jstat = list(filter(None.__ne__, jstat))

            jmean = np.mean(jstat, axis=0)
            a = np.sum((jmean - jstat)**3, axis=0) / \
                (6.0 * np.sum((jmean - jstat)**2, axis=0)**(3 / 2))
            zs = z0 + \
                norm.ppf(alphas).reshape(alphas.shape + (1,) * z0.ndim)

            avals = norm.cdf(z0 + zs / (1 - a * zs))

            nvals = np.round((len(current_) - 1) * avals)
            nvals = np.nan_to_num(nvals).astype('int')

            low_conf = np.zeros(shape=(len(current_[0]), len(current_[0])))
            high_conf = np.zeros(
                shape=(len(current_[0]), len(current_[0])))

            for i in range(len(current_[0])):
                for j in range(len(current_[0])):
                    low_conf[i][j] = (current_[nvals[0][i][j]][i][j])

            for i in range(len(*current[0])):
                for j in range(len(*current[0])):
                    high_conf[i][j] = (current_[nvals[1][i][j]][i][j])

            print('MEAN')
            print(np.round(np.mean(current_, axis=0), 4))
            print('CI LOW')
            print(avals[0])
            print(low_conf)
            print('CI HIGH')
            print(avals[1])
            print(high_conf)
            print('t-value')
            tstat = np.nan_to_num(np.mean(current_, axis=0) /
                                  np.std(current_, axis=0, ddof=1))
            print(tstat)
