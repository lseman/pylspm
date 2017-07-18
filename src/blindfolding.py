# CHIN, W. W. How to Write Up and Report PLS Analyses. In: Handbook of
# Partial Least Squares. Berlin, Heidelberg: Springer Berlin Heidelberg,
# 2010. p. 655–690.

import pandas
import numpy as np
from numpy import inf
import pandas as pd

from pylspm import PyLSpm
from boot import PyLSboot


def isNaN(num):
    return num != num


def blindfolding(data_, lvmodel, mvmodel, scheme,
                 regression, h='0', maxit='100', HOC='true'):

    model = PyLSpm(data_, lvmodel, mvmodel, scheme,
                   regression, h, maxit, HOC=HOC)
    data2_ = model.data
    # observation/distance must not be interger
    distance = 7
    Q2 = pd.DataFrame(0, index=data2_.columns.values,
                      columns=range(distance))

    SSE = pd.DataFrame(0, index=data2_.columns.values,
                       columns=range(distance))

    SSO = pd.DataFrame(0, index=data2_.columns.values,
                       columns=range(distance))

    mean = pd.DataFrame.mean(data2_)

    for dist in range(distance):
        dataBlind = data_.copy()
        rodada = 1
        count = distance - dist - 1
        for j in range(len(data_.columns)):
            for i in range(len(data_)):
                count += 1
                if count == distance:
                    dataBlind.ix[i, j] = np.nan
                    count = 0

        for j in range(len(data_.columns)):
            for i in range(len(data_)):
                if (isNaN(dataBlind.ix[i, j])):
                    dataBlind.ix[i, j] = mean[j]

        rodada = rodada + 1

        plsRound = PyLSpm(dataBlind, lvmodel, mvmodel,
                          scheme, regression, 0, 100, HOC='true')
        predictedRound = plsRound.predict()

        SSE[dist] = pd.DataFrame.sum((data2_ - predictedRound)**2)
        SSO[dist] = pd.DataFrame.sum((data2_ - mean)**2)

    latent = plsRound.latent
    Variables = plsRound.Variables

    SSE = pd.DataFrame.sum(SSE, axis=1)
    SSO = pd.DataFrame.sum(SSO, axis=1)

    Q2latent = pd.DataFrame(0, index=np.arange(1), columns=latent)

    for i in range(len(latent)):
        block = data2_[Variables['measurement'][
            Variables['latent'] == latent[i]]]
        block = block.columns.values

        SSEblock = pd.DataFrame.sum(SSE[block])
        SSOblock = pd.DataFrame.sum(SSO[block])

        Q2latent[latent[i]] = 1 - (SSEblock / SSOblock)

    Q2 = 1 - (SSE / SSO)
    print(Q2)
    Q2latent = Q2latent.T
    print(Q2latent)
