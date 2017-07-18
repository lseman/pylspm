# PyLS-PM bootstraping MPI Library
# Author: Laio Oriel Seman
# Creation: Jun 2017

from multiprocessing import Pool, freeze_support

from mpi4py import MPI
import sys

import pandas as pd
import numpy as np
from .pylspm import PyLSpm
import random
from scipy.stats.stats import pearsonr
from .boot import PyLSboot


def do_work(data, LVcsv, Mcsv, scheme, reg, h, maximo):
    amostra = data.sample(
        len(data), replace=True, random_state=(np.random.RandomState()))
    amostra.index = range(len(data))
    try:
        bootstraping = PyLSpm(amostra, LVcsv, Mcsv, scheme,
                              reg, h, maximo)
        if (bootstraping.convergiu == 1):
            return [bootstraping.path_matrix.values]
    except:
        return None


def do_work_pso(data, LVcsv, Mcsv, scheme, reg, h, maximo):
    output = pd.DataFrame(population[item].position)
    output.columns = ['Split']
    dataSplit = pd.concat([data, output], axis=1)
    f1 = []
    results = []
    for i in range(nclusters):
        dataSplited = (dataSplit.loc[dataSplit['Split']
                                     == i]).drop('Split', axis=1)
        dataSplited.index = range(len(dataSplited))

        try:
            results.append(PyLSpm(dataSplited, LVcsv, Mcsv, scheme,
                                  reg, 0, 50, HOC='true'))

            resid = results[i].residuals()[3]
            f1.append(resid)
        except:
            f1.append(10000)
    print((1 / np.sum(f1)))
    return (1 / np.sum(f1))

def PyLSmpi(br, cores, dados, LVcsv, Mcsv, scheme='path', reg='ols', h=0, maximo=300, stopCrit=7, nclusters=2, population=None):

    data = dados
    LVcsv = LVcsv
    Mcsv = Mcsv
    maximo = maximo
    stopCriterion = stopCrit
    cores = cores
    scheme = scheme
    reg = reg
    nboot = br

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mode = 'pso'
    if mode == 'bootstrap':
        mode_ = 0
    elif mode == 'pso':
        mode_ = 1

    if rank == 0:
        nboot = (nboot / size) + (nboot % size)
    else:
        nboot = (nboot / size)

    data = 'dados_missForest.csv'
    lvmodel = 'lvnew.csv'
    mvmodel = 'mvnew.csv'
    scheme = 'path'
    regression = 'ols'
    algorithm = 'wold'

    data = pd.read_csv(data)

    result = []
    for i in range(int(nboot)):
        result_ = do_work(data, lvmodel, mvmodel, scheme, regression, 0, 100)
        result.append(result_)

    if rank == 0:
        current = result
        for i in range(size - 1):
            datinha = comm.recv(source=i + 1)
            current = np.concatenate((current, datinha), axis=0)
    else:
        comm.send(result, 0)

    if comm.rank == 0:
        current = list(filter(None.__ne__, current))
        current = np.sort(current, axis=0)

        print(len(current))
        print(current)

        '''print('MEAN')
        print(np.round(np.mean(current, axis=0), 4))
        print('STD')
        print(np.round(np.std(current, axis=0, ddof=1), 4))
        print('CI 2.5')
        print(np.round(np.percentile(current, 2.5, axis=0), 4))
        print('CI 97.5')
        print(np.round(np.percentile(current, 97.5, axis=0), 4))
        print(size)
        print(sys.argv[0])'''