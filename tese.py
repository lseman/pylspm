from multiprocessing import Pool, freeze_support

import pandas
import numpy as np
from numpy import inf
import pandas as pd
import scipy.stats
from scipy.stats import norm

import matplotlib
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, ward, distance
from scipy.cluster.hierarchy import fcluster

from pylspm import PyLSpm
from results import PyLSpmHTML
from boot import PyLSboot
from rebus import rebus
from blindfolding import blindfolding

import csv

if __name__ == '__main__':
    freeze_support()

    def print_full(x):
        pd.set_option('display.max_columns', len(x))
        print(x)
        pd.reset_option('display.max_columns')

    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 10}

    matplotlib.rc('font', **font)

    # Parâmetros

    boot = 0
    nrboot = 100
    cores = 8

    diff = 'cluster'
    method = 'percentile'
    data = 'dados_reag.csv'
    lvmodel = 'lvnew.csv'
    mvmodel = 'mvnew.csv'
    scheme = 'path'
    regression = 'ols'

    # Trata missing data (mean)

    def isNaN(num):
        return num != num

    data_ = pandas.read_csv(data)
    mean = pd.DataFrame.mean(data_)
    for j in range(len(data_.columns)):
        for i in range(len(data_)):
            if (isNaN(data_.ix[i, j])):
                data_.ix[i, j] = mean[j]

    # Go!

    if (boot == 0):

        tese = PyLSpm(data_, lvmodel, mvmodel, scheme,
                      regression, 0, 100, HOC='true')

        if (diff == 'sample'):
            tese.sampleSize()

            plt.plot(tese.sampleSize()[0], tese.sampleSize()[1], 'o-')
            plt.xlabel('Potência')
            plt.ylabel('Tamanho da Amostra')
            plt.grid(True)
            plt.show()

        if (diff == 'cluster'):
            rebus(tese.residuals()[0], data_, tese.data, lvmodel, mvmodel, scheme, regression)

        imprime = PyLSpmHTML(tese)
        imprime.generate()

    elif (boot == 1):
        tese = PyLSboot(nrboot, cores, data_, lvmodel,
                        mvmodel, scheme, regression, 0, 100)
        resultados = tese.boot()

        default = PyLSpm(data_, lvmodel, mvmodel, scheme,
                         regression, 0, 100).path_matrix.values

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
                pvalue = (scipy.stats.t.sf(
                    tstat, len(current_) - 1))
                print(pvalue)

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

    elif (boot == 2):
        blindfolding(data_, lvmodel, mvmodel, scheme,
                  regression, 0, 100, HOC='true')

    elif (boot == 3):

        def trataGroups(objeto):

            current = list(filter(None.__ne__, objeto))
            current = np.sort(current, axis=0)

            for i in range(len(current[0])):
                current_ = [j[i] for j in current]
                mean_ = np.round(np.mean(current_, axis=0), 4)
                deviation_ = np.round(np.std(current_, axis=0, ddof=1), 4)

            return [mean_, deviation_]

        data1 = (data_.loc[data_['SEM'] == 0]).drop('SEM', axis=1)
        data2 = (data_.loc[data_['SEM'] == 1]).drop('SEM', axis=1)

        estimData1 = PyLSboot(nrboot, cores, data1, lvmodel,
                              mvmodel, scheme, regression, 0, 100)
        estimData2 = PyLSboot(nrboot, cores, data2, lvmodel,
                              mvmodel, scheme, regression, 0, 100)

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

        method = 'non-parametric'

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

        print(tStat)
        print(pval)
        print(df)

        print('Paths')
        print(estimado1[0])
        print(estimado2[0])

    elif (boot == 4):

        data1 = (data_.loc[data_['SEM'] == 0]).drop('SEM', axis=1).values
        data2 = (data_.loc[data_['SEM'] == 1]).drop('SEM', axis=1).values

        print(data1)

        levene = scipy.stats.levene(data1, data2)
