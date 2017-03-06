# PLS-PM genetic algorithm clustering
# Author: Laio Oriel Seman

# Demasi, P. Heurísticas Baseadas em Apostas para Problemas de Otimização
# Combinatória. Tese (Doutorado em Informática) – Universidade Federal do
# Riode Janeiro, Instituto de Matemática, Programa de Pós-Graduação em
# Informática, Rio de Janeiro, 2015.

from random import randint, uniform
from copy import deepcopy
from sys import argv
import numpy as np
from numpy import inf
import sys
import pandas as pd
import random

from pylspm import PyLSpm
from boot import PyLSboot

sys.setrecursionlimit(1500)


def BinSearch(prob, p, imin, imax):
    imid = (imin + imax) / 2
    if imid == len(prob) - 1 or imid == 0:
        return imid
    if p > prob[int(imid)] and p <= prob[int(imid) + 1]:
        return imid + 1
    elif p < prob[int(imid)]:
        imid = BinSearch(prob, p, imin, imid)
    else:
        imid = BinSearch(prob, p, imid, imax)
    return imid


class Individual(object):

    def __init__(self, data_, n_clusters, playes):
        self.playes = playes
        if not self.playes:
            for i in range(len(data_)):
                self.playes.append(random.randrange(n_clusters))

    def fitness(self, data_, n_clusters, lvmodel, mvmodel, scheme, regression):

        output = pd.DataFrame(self.playes)
        output.columns = ['Split']
        dataSplit = pd.concat([data_, output], axis=1)
        f1 = []
        results = []
        for i in range(n_clusters):
            dataSplited = (dataSplit.loc[dataSplit['Split']
                                         == i]).drop('Split', axis=1)
            dataSplited.index = range(len(dataSplited))

            try:
                results.append(PyLSpm(dataSplited, lvmodel, mvmodel, scheme,
                                      regression, 0, 50, HOC='true'))

                sumOuterResid = pd.DataFrame.sum(
                    pd.DataFrame.sum(results[i].residuals()[1]**2))
                sumInnerResid = pd.DataFrame.sum(
                    pd.DataFrame.sum(results[i].residuals()[2]**2))
                f1.append(sumOuterResid + sumInnerResid)
            except:
                f1.append(10000)

        print((1 / np.sum(f1)))
        return (1 / np.sum(f1))

def initPopulation(npop, data_, n_clusters):
    return [Individual(data_, n_clusters, []) for i in range(0, npop)]

def bac(npop, n_clusters, pcros, pmut, maxit, data_,
        lvmodel, mvmodel, scheme, regression):

    pop = initPopulation(npop, data_, n_clusters)
    bestfit = [0, 0]

    for i in range(0, maxit):
        print("Iteration %s" % (i + 1))

        # Create mask
        mask = []
        for j in range(len(data_)):
            mask.append(random.randrange(n_clusters))

        # Possible solutions
        print('Possible solutions')
        print(n_clusters**len(data_))

        # Calculate fitness
        fit_ = PyLSboot(len(pop), 8, data_, lvmodel,
                        mvmodel, scheme, regression, 0, 100, nclusters=n_clusters, population=pop)
        fit = fit_.gac()
        # calcular w

        m = ?
        for j in range(m):
            w = atualiza

        for j in range(len(pop)):
            for k in range():
                p =
                w = 
                B = 
                aposte B



        pop = deepcopy(new)

        # elitism
        if max(fit) > bestfit[1]:
            bestfit = [pop[np.argmax(fit)], max(fit)]

    print("\nFitness = %s" % bestfit[1])
    print(bestfit[0].playes)

    output = pd.DataFrame(bestfit[0].playes)
    output.columns = ['Split']
    dataSplit = pd.concat([data_, output], axis=1)

    # Return best clusters path matrix
    results = []
    for i in range(n_clusters):
        dataSplited = (dataSplit.loc[dataSplit['Split']
                                     == i]).drop('Split', axis=1)
        dataSplited.index = range(len(dataSplited))
        results.append(PyLSpm(dataSplited, lvmodel, mvmodel, scheme,
                              regression, 0, 100, HOC='true'))

        print(results[i].path_matrix)
        print(results[i].gof())
