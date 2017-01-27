# L. Trinchera, “Unobserved Heterogeneity in Structural Equation Models: A
# New Approach to Latent Class Detection in PLS Path Modeling,” 2007.

from multiprocessing import Pool, freeze_support

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
from mga import mga
from itertools import combinations


def rebus(residuals, data, dataRealoc, lvmodel, mvmodel, scheme, regression):

    Z = linkage(residuals, method='ward')
    plt.figure(figsize=(15, 8))
    plt.title('Dendograma de Agrupamento Hierárquico')
    plt.xlabel('Amostra')
    plt.ylabel('Distância')
    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=8,
    )
    plt.show()
    max_d = 17
    clusters = fcluster(Z, max_d, criterion='distance')

    while True:
        clusters = pd.DataFrame(clusters)
        clusters.columns = ['Split']

        old_clusters = clusters.copy()

        dataSplit = pd.concat([data, clusters], axis=1)

        nk = max(clusters['Split'])

        rebus = []
        for i in range(nk):
            data_ = (dataSplit.loc[dataSplit['Split']
                                   == i + 1]).drop('Split', axis=1)
            data_.index = range(len(data_))
            rebus.append(PyLSpm(data_, lvmodel, mvmodel, scheme,
                                regression, 0, 100, HOC='true'))

        CM = pd.DataFrame(0, index=np.arange(len(data)), columns=np.arange(nk))

        exoVar = rebus[i].endoexo()[1]
        endoVar = rebus[i].endoexo()[0]

        for j in range(nk):

            dataRealoc_ = dataRealoc.copy()

            # Novos residuais

            mean_ = np.mean(rebus[j].data, 0)
            scale_ = np.std(rebus[j].data, 0) * \
                np.sqrt((len(data_) - 1) / len(data_))

            dataRealoc_ = dataRealoc_ - mean_
            dataRealoc_ = dataRealoc_ / scale_

            outer_residuals = dataRealoc_.copy()
            fscores = pd.DataFrame.dot(dataRealoc_, rebus[0].outer_weights)

            for i in range(len(rebus[j].latent)):
                block = dataRealoc_[rebus[j].Variables['measurement']
                                    [rebus[j].Variables['latent'] == rebus[j].latent[i]]]
                block = block.columns.values

                loadings = rebus[j].outer_loadings.ix[
                    block][rebus[j].latent[i]].values

                outer_ = fscores.ix[:, i].values
                outer_ = outer_.reshape(len(outer_), 1)
                loadings = loadings.reshape(len(loadings), 1)
                outer_ = np.dot(outer_, loadings.T)

                outer_residuals.ix[:, block] = (dataRealoc_.ix[
                    :, block] - outer_)**2

                inner_residuals = fscores[endoVar]
                inner_ = pd.DataFrame.dot(
                    fscores, rebus[j].path_matrix.ix[endoVar].T)
                inner_residuals = (fscores[endoVar] - inner_)**2

            # Fim dos novos residuais

            resnum1 = pd.DataFrame.dot(outer_residuals, (np.diag(
                1 / (pd.DataFrame.sum(
                    rebus[j].comunalidades(), axis=1)).values.flatten())))
            supresouter = pd.DataFrame.sum(
                resnum1, axis=1) / (pd.DataFrame.sum(pd.DataFrame.sum(resnum1, axis=1)) / (len(data) - 2))

            resnum2 = pd.DataFrame.dot(inner_residuals, (np.diag(
                1 / rebus[j].r2.ix[endoVar].values.flatten())))
            supresinner = pd.DataFrame.sum(
                resnum2, axis=1) / (pd.DataFrame.sum(pd.DataFrame.sum(resnum2, axis=1)) / (len(data) - 2))

            CM.ix[:, j] = (np.sqrt(supresouter * supresinner))

        clusters = CM.idxmin(axis=1).values
        clusters = clusters + 1

        diff_clusters = clusters - old_clusters.values.flatten()

        changes = diff_clusters.astype(bool).sum()

        print(changes)
        if((changes / len(data)) < 0.005):
            break

        old_clusters = clusters.copy()

    # Estima final

    clusters = pd.DataFrame(clusters)
    clusters.columns = ['Split']

    dataSplit = pd.concat([data, clusters], axis=1)

    nk = max(clusters['Split'])
    rebus = []
    for i in range(nk):
        data_ = (dataSplit.loc[dataSplit['Split']
                               == i + 1]).drop('Split', axis=1)
        data_.index = range(len(data_))
        rebus.append(PyLSpm(data_, lvmodel, mvmodel, scheme,
                            regression, 0, 100, HOC='true'))
        print(np.round(len(data_) / len(data) * 100, 2))
        print(len(data_))
        print(rebus[i].path_matrix)
        print(rebus[i].gof())

    # Automatiza multi-group

    allCombs = list(combinations(range(1, nk + 1), 2))

    for i in range(len(allCombs)):
        mga(50, 8, dataSplit, lvmodel,
            mvmodel, scheme, regression, 0, 100, g1=allCombs[i][0], g2=allCombs[i][1],
            segmento='Split')
