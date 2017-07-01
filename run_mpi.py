from mpi4py import MPI
import sys

import pandas as pd
import numpy as np
from pylspm import PyLSpm
import random
from scipy.stats.stats import pearsonr
from boot import PyLSboot


# Functions

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


def do_work_pso(item, nclusters, data, LVcsv, Mcsv, scheme, reg, h, maximo, population):
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
#    print((1 / np.sum(f1)))
    return (1 / np.sum(f1))


def do_work_ga(item, nclusters, data, LVcsv, Mcsv, scheme, reg, h, maximo, population):
    output = pd.DataFrame(population[item].genes)
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
    return (1 / np.sum(f1))

# Main

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

initcomm = MPI.Comm.Get_parent()

# Data Config

lvmodel = 'lvnew.csv'
mvmodel = 'mvnew.csv'
scheme = 'path'
regression = 'ols'
algorithm = 'wold'

received = initcomm.recv(source=0)
nboot = received[0]
mode = received[1]
population = received[2]
data = received[3]

#data = pd.read_csv(data)

# MODES

result = []
if mode == 0:
    for i in range(int(nboot)):
        result_ = do_work(data, lvmodel, mvmodel, scheme, regression, 0, 100)
        result.append(result_)

elif mode == 1:
    item = rank + int(nboot) - 1
    for i in range(int(nboot)):
        result_ = do_work_pso(item, 3, data, lvmodel,
                              mvmodel, scheme, regression, 0, 100, population)
        result.append(result_)
        item += 1

elif mode == 2:
    item = rank + int(nboot) - 1
    for i in range(int(nboot)):
        result_ = do_work_ga(item, 3, data, lvmodel,
                              mvmodel, scheme, regression, 0, 100, population)
        result.append(result_)
        item += 1

######

if rank == 0:
    current = result
    for i in range(size - 1):
        datinha = comm.recv(source=i + 1)
        current = np.concatenate((current, datinha), axis=0)
    initcomm.send(current, 0)
else:
    comm.send(result, 0)

initcomm.Disconnect()
