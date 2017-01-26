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

    mode = 4
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

        imprime = PyLSpmHTML(tese)
        imprime.generate()

    elif (mode == 1):
        bootstrap(nrboot, cores, data_, lvmodel,
                  mvmodel, scheme, regression, 0, 100)

    elif (mode == 2):
        blindfolding(data_, lvmodel, mvmodel, scheme,
                     regression, 0, 100, HOC='true')

    elif (mode == 3):
        mga(nrboot, cores, data_, lvmodel,
            mvmodel, scheme, regression, 0, 100, g1=0, g2=1)

    elif (mode == 4):
        n_individuals = 3
        n_clusters = 3
        p_crossover = 0.6
        p_mutation = 1-((0.3)**(1.0/(1.0*len(data_))))
        print(p_mutation)
        iterations = 5

        gac(n_individuals, n_clusters,
            p_crossover, p_mutation, iterations,
            data_, lvmodel, mvmodel, scheme, regression)