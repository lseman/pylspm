from multiprocessing import Pool, freeze_support

import numpy as np
from numpy import inf
import pandas as pd
import scipy.stats
from scipy.stats import norm

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
#from bac import bac

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

    mode = 5
    nrboot = 10
    cores = 8

    diff = 'none'
    method = 'percentile'
    data = 'dados_miss.csv'
    lvmodel = 'lvnew.csv'
    mvmodel = 'mvnew.csv'
    scheme = 'path'
    regression = 'ols'

    # Trata missing data (mean)

    def isNaN(num):
        return num != num

    data_ = pd.read_csv(data)
    mean = pd.DataFrame.mean(data_)
    for j in range(len(data_.columns)):
        for i in range(len(data_)):
            if (isNaN(data_.ix[i, j])):
                data_.ix[i, j] = mean[j]

    # Go!

#    data_ = data_.drop('SEM', axis=1)

    if (mode == 0):

        tese = PyLSpm(data_, lvmodel, mvmodel, scheme,
                      regression, 0, 100, HOC='true')

        if (diff == 'sample'):
            tese.sampleSize()

            plt.plot(tese.sampleSize()[0], tese.sampleSize()[1], 'o-')
            plt.xlabel('Potência')
            plt.ylabel('Tamanho da Amostra')
            plt.grid(True)
            plt.show()

        elif (diff == 'rebus'):
            rebus(tese.residuals()[0], data_, tese.data,
                  lvmodel, mvmodel, scheme, regression)

#        print(tese.path_matrix)
        print(tese.residuals()[3])

        imprime = PyLSpmHTML(tese)
        imprime.generate()

    # Bootstrap
    elif (mode == 1):
        bootstrap(nrboot, cores, data_, lvmodel,
                  mvmodel, scheme, regression, 0, 100)

    # Blindfolding
    elif (mode == 2):
        blindfolding(data_, lvmodel, mvmodel, scheme,
                     regression, 0, 100, HOC='true')

    # Multigroup Analysis
    elif (mode == 3):
        mga(nrboot, cores, data_, lvmodel,
            mvmodel, scheme, regression, 0, 100, g1=0, g2=1)

    # Genetic Algorithm
    elif (mode == 4):
        n_individuals = 20
        n_clusters = 3
        p_crossover = 0.85
#        p_mutation = 1-((0.3)**(1.0/(1.0*len(data_))))
#        print(p_mutation)
        p_mutation = 0.01
        iterations = 100

        gac(n_individuals, n_clusters,
            p_crossover, p_mutation, iterations,
            data_, lvmodel, mvmodel, scheme, regression)

    # PSO
    elif (mode == 5):
        n_individuals = 3
        n_clusters = 3
        in_max = 0.9
        in_min = 0.5
        c1 = 1.5
        c2 = 1.5
        iterations = 3

        pso(n_individuals, n_clusters,
            in_max, in_min, c1, c2, iterations,
            data_, lvmodel, mvmodel, scheme, regression)

    elif (mode == 6):
        n_players = 4
        n_clusters = 3
        p_crossover = 0.85
        p_mutation = 0.01
        iterations = 3

        bac(n_players, n_clusters,
            p_crossover, p_mutation, iterations,
            data_, lvmodel, mvmodel, scheme, regression)
