from multiprocessing import Pool, freeze_support

import numpy as np
from numpy import inf
import pandas as pd
import scipy.stats
from scipy.stats import norm
import itertools

import matplotlib
from matplotlib import pyplot as plt

from pylspm import PyLSpm
from results import PyLSpmHTML
from boot import PyLSboot

from rebus import rebus
from blindfolding import blindfolding
from bootstraping import bootstrap
from mga import mga
from gac import gac
from pso import pso
from tabu2 import tabu
from permuta import permuta
from plsr2 import plsr2, HOCcat
from monteCholesky import monteCholesky
from adequacy import *

if __name__ == '__main__':
    freeze_support()

    pd.set_option('display.expand_frame_repr', False)

    def print_full(x):
        pd.set_option('display.max_columns', len(x))
        print(x)
        pd.reset_option('display.max_columns')

    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 10}

    matplotlib.rc('font', **font)
    matplotlib.style.use('ggplot')

    # Parâmetros

    mode = 0
    nrboot = 500
    cores = 8
    nrepic = 500

    diff = 'none'
    method = 'percentile'
    data = 'dados_missForest.csv'
    lvmodel = 'lvnew.csv'
    mvmodel = 'mvnew.csv'
    scheme = 'path'
    regression = 'ols'
    algorithm = 'wold'

    def isNaN(num):
        return num != num

    data_ = pd.read_csv(data)

    # Mean replacement

    """mean = pd.DataFrame.mean(data_)
    for j in range(len(data_.columns)):
        for i in range(len(data_)):
            if (isNaN(data_.ix[i, j])):
                data_.ix[i, j] = mean[j]"""

#    data_ = data_.drop('SEM', axis=1)

#    g1 = 4
#    segmento = 'SEM'
#    data_ = (data_.loc[data_[segmento] == g1]).drop(segmento, axis=1)
#    print(data_)

#    data_, mvmodel = HOCcat(data_, mvmodel, seed=9002)

    if (mode == 0):

        tese = PyLSpm(data_, lvmodel, mvmodel, scheme,
                      regression, 0, 100, HOC='false', disattenuate='false', method=algorithm)

        if (diff == 'sample'):
            tese.sampleSize()

            plt.plot(tese.sampleSize()[0], tese.sampleSize()[1], '-')
            plt.xlabel('Potência')
            plt.ylabel('Tamanho da Amostra')
            plt.grid(True)
            plt.show()

        elif (diff == 'rebus'):
            rebus(tese.residuals()[0], data_, tese.data,
                  lvmodel, mvmodel, scheme, regression)

        elif (diff == 'plot'):

            tese.PCA()
            SEM = data_['SEM']
            tese.frequencyPlot(data_, SEM)
            tese.impa()
            tese.scatterMatrix()

            data_ = data_.drop('SEM', axis=1)
            data_.boxplot(grid=False)
            plt.savefig('imgs/boxplot', bbox_inches='tight')
            plt.clf()
            plt.cla()

        imprime = PyLSpmHTML(tese)
        imprime.generate()

        tese.impa()

        print(tese.path_matrix)

    # Monte Carlo with Cholesky
    elif (mode == 10):
        monteCholesky(nrepic, nrboot, cores, data_, lvmodel,
                      mvmodel, scheme, regression, 0, 100)

    # Bootstrap
    elif (mode == 1):
        bootstrap(nrboot, cores, data_, lvmodel,
                  mvmodel, scheme, regression, 0, 100, method)

    # Blindfolding
    elif (mode == 2):
        blindfolding(data_, lvmodel, mvmodel, scheme,
                     regression, 0, 100, HOC='true')

    # Multigroup Analysis
    elif (mode == 3):
#        mga(nrboot, cores, data_, lvmodel,
#                mvmodel, scheme, regression, 0, 100, g1=0, g2=3)

        comb = list(itertools.combinations(range(6), 2))
        for k,j in comb:
            print("Groups " + str(k+1) + '-' + str(j+1))
            mga(nrboot, cores, data_, lvmodel,
                mvmodel, scheme, regression, 0, 100, g1=k, g2=j)

    # Permutation
    elif (mode == 4):
#        permuta(nrboot, cores, data_, lvmodel,
#                mvmodel, scheme, regression, 0, 100, g1=3, g2=4)

        comb = list(itertools.combinations(range(6), 2))
        for k,j in comb:
            print("Groups " + str(k+1) + '-' + str(j+1))
            permuta(nrboot, cores, data_, lvmodel,
                    mvmodel, scheme, regression, 0, 100, g1=k, g2=j)

    # GA
    elif (mode == 5):
        n_individuals = 10
        n_clusters = 4
        p_crossover = 0.85
#        p_mutation = 1-((0.3)**(1.0/(1.0*len(data_))))
#        print(p_mutation)
        p_mutation = 0.03
        iterations = 100

        gac(n_individuals, n_clusters,
            p_crossover, p_mutation, iterations,
            data_, lvmodel, mvmodel, scheme, regression)

    # PSO
    elif (mode == 6):
        n_individuals = 5
        n_clusters = 3
        in_max = 0.9
        in_min = 0.5
        c1 = 1.5
        c2 = 1.5
        iterations = 100

        pso(n_individuals, n_clusters,
            in_max, in_min, c1, c2, iterations,
            data_, lvmodel, mvmodel, scheme, regression)

    # TS
    elif (mode == 7):
        tabu_size = 10
        n_children = 3
        n_clusters = 3
        iterations = 100

        tabu(tabu_size, n_children, n_clusters, iterations,
             data_, lvmodel, mvmodel, scheme, regression)
