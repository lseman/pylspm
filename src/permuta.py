# HENSELER, J.; RINGLE, C. M.; SARSTEDT, M. Testing measurement invariance
# of composites using partial least squares. International Marketing
# Review, v. 33, n. 3, p. 405–431, 9 maio 2016.

import pandas
import numpy as np
from numpy import inf
import pandas as pd
import scipy.stats
from scipy.stats import norm

from pylspm import PyLSpm
from boot import PyLSboot
from itertools import combinations
import itertools
import random


def trataGroups(objeto):

    current = list(filter(None.__ne__, objeto))

    mean_ = np.round(np.mean(current, axis=0), 4)
    deviation_ = np.round(np.std(current, axis=0, ddof=1), 4)
    fivecent = np.round(np.percentile(current, 5.0, axis=0), 4)

    # confidence intervals
    lowci = np.round(np.percentile(current, 2.5, axis=0), 4)
    highci = np.round(np.percentile(current, 97.5, axis=0), 4)

    return [mean_, deviation_, fivecent, current, lowci, highci]


def permuta(nrboot, cores, data_, lvmodel,
            mvmodel, scheme, regression, h='0', maxit='100', g1=0, g2=1, segmento='SEM'):

    dataPermuta = (data_.loc[(data_[segmento] == g1) | (
        data_[segmento] == g2)]).drop(segmento, axis=1)

    # Original
    dataOrign = []
    for i in range(2):
        dataOrign.append(
            (data_.loc[data_[segmento] == i]).drop(segmento, axis=1))

    estimaOrign = []
    f1 = []

    for i in range(2):
        estimaOrign.append(PyLSpm(dataOrign[i], lvmodel, mvmodel, scheme,
                                  regression, 0, 100, HOC='false'))
        outer_weights = estimaOrign[i].outer_weights
        f1.append(outer_weights)

    score1 = pd.DataFrame.dot(estimaOrign[0].normaliza(dataPermuta), f1[0])
    score2 = pd.DataFrame.dot(estimaOrign[0].normaliza(dataPermuta), f1[1])
    c = []
    for i in range(len(score1.columns)):
        c_ = np.corrcoef(score1.ix[:, i], score2.ix[:, i])
        c.append(c_[0][1])

    print('Original C')
    print(np.round(c, 4))

    # Permutation
    estimData = PyLSboot(nrboot, cores, data_, lvmodel,
                         mvmodel, scheme, regression, h, maxit, g1=g1, g2=g2, segmento=segmento)

    estimData = estimData.permuta()
    estimado = trataGroups(estimData)

    # Step 2 n Step 3
    '''print('c Mean')
    print(estimado[0][0])
    print('c 5%')
    print(estimado[2][0])

    print('diff Mean')
    print(estimado[0][1])
    print('log diff Mean')
    print(estimado[0][2])

    print('LOW CI')
    print('h2')
    print(estimado[4][1])
    print('h3')
    print(estimado[4][2])

    print('HIGH CI')
    print('h2')
    print(estimado[5][1])
    print('h3')
    print(estimado[5][2])

    print('p-value')
    pval = np.sum([np.array(c) < np.array(estimado[3])],
                  axis=1) * (1 / (nrboot + 1))
    print(pval[0][0])'''

    print('test Compositional Invariance')
    print(np.round(c, 4) < estimado[2][0])

#    print('Equality of Composite Mean Values and Variance')
#    print(estimado[0][1]>estimado[4][1])
#    print(estimado[0][1]<estimado[5][1])
#    print(estimado[0][2]>estimado[4][2])
#    print(estimado[0][2]<estimado[5][2])
