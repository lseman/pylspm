import numpy as np
from numpy import inf
import pandas as pd
import scipy.stats
from pylspm import PyLSpm
from results import PyLSpmHTML
from boot import PyLSboot
from itertools import combinations
from mga import mga
from permuta import permuta


def test_heuristic(nrboot, cores, data_, lvmodel, mvmodel, scheme, regression, h='0', maxit='100'):

    test = 'pso'

    if test == 'ga':
        split = [1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
                 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1]
    if test == 'pso':
        split = [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0,
                 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0]

    split = pd.DataFrame(split)
    split.columns = ['Split']
    dataSplit = pd.concat([data_, split], axis=1)
    nk = max(split['Split'])

    splitado = []
    f1 = []
    for i in range(nk + 1):
        data_ = (dataSplit.loc[dataSplit['Split']
                               == i]).drop('Split', axis=1)
        data_.index = range(len(data_))
        estima = PyLSpm(data_, lvmodel, mvmodel, scheme,
                        regression, 0, 100, HOC='true')
        print(estima.path_matrix)
        f1.append(estima.residuals()[3])
    print(f1)
    print(np.sum(f1))
    print(1 / np.sum(f1))

    compara = 1

    if compara == 1:
        allCombs = list(combinations(range(0, nk + 1), 2))
        for i in range(len(allCombs)):
            print("Groups " + str(allCombs[i][0]) + '-' + str(allCombs[i][1]))
            print('MGA')
            mga(50, 8, dataSplit, lvmodel,
                mvmodel, scheme, regression, 0, 100, g1=allCombs[i][0], g2=allCombs[i][1],
                segmento='Split')
            print('Permutation')
            permuta(nrboot, cores, data_, lvmodel,
                    mvmodel, scheme, regression, 0, 100, g1=allCombs[i][0], g2=allCombs[i][1])
