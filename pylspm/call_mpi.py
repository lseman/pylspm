from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import pandas as pd
import numpy as np
from .pylspm import PyLSpm
import random
from scipy.stats.stats import pearsonr
from .boot import PyLSboot


def PyLSmpi(mode, br, cores, dados, LVcsv, Mcsv, scheme='path', reg='ols', h=0, maximo=300, stopCrit=7, nclusters=2, population=None):

    data = dados
    LVcsv = LVcsv
    Mcsv = Mcsv
    maximo = maximo
    stopCriterion = stopCrit
    cores = cores
    scheme = scheme
    reg = reg

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpisize = comm.Get_size()

    mode_ = 0
    if mode == 'bootstrap':
        mode_ = 0
    elif mode == 'pso':
        mode_ = 1
    elif mode == 'ga':
        mode_ = 2

    nboot = []
    nboot.append((br / cores) + (br % cores))
    for i in range(cores-1):
        nboot.append(br / cores)

    myinfo = MPI.Info.Create()
    myinfo.Set("hostfile", "machinefile")
    initcomm = MPI.COMM_SELF.Spawn(sys.executable, args=['run_mpi.py'], maxprocs=cores)

    for i in range(cores):
        initcomm.send([nboot[i], mode_, population, data], i)

    current = None
    current = initcomm.recv(current)
#    current = list(filter(None.__ne__, current))
#    current = np.sort(current, axis=0)

#    print(len(current))
#    print('MEAN')
#    mean = np.round(np.mean(current, axis=0), 4)
#    print(mean)

    initcomm.Disconnect()

    return current