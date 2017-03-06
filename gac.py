# PLS-PM genetic algorithm clustering
# Author: Laio Oriel Seman

# C. M. Ringle, M. Sarstedt, R. Schlittgen, and C. R. Taylor, “PLS path
# modeling and evolutionary segmentation,” J. Bus. Res., vol. 66, no. 9,
# pp. 1318–1324, Sep. 2013.

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

    def __init__(self, data_, n_clusters, genes):
        self.genes = genes
        if not self.genes:
            for i in range(len(data_)):
                self.genes.append(random.randrange(n_clusters))

    def fitness(self, data_, n_clusters, lvmodel, mvmodel, scheme, regression):

        output = pd.DataFrame(self.genes)
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

    def mutation(self, pmut, n_clusters):
        for g, gene in enumerate(self.genes):
            if uniform(0, 1) <= pmut:
                oldgene = self.genes[g]
                # garante mutation diferente
                while self.genes[g] == oldgene:
                    self.genes[g] = random.randrange(n_clusters)


def initPopulation(npop, data_, n_clusters):
    return [Individual(data_, n_clusters, []) for i in range(0, npop)]


def crossover(parent1, parent2, n_clusters):
    point = randint(1, len(parent1.genes) - 2)
    return Individual(None, n_clusters, parent1.genes[:point] + parent2.genes[point:]), Individual(None, n_clusters, parent2.genes[:point] + parent1.genes[point:])


def roulettewheel(pop, fit):
    fit = fit - min(fit)
    sumf = sum(fit)
    if(sumf == 0):
        return pop[0]
    prob = [(item + sum(fit[:index])) / sumf for index, item in enumerate(fit)]
    prob_ = uniform(0, 1)
#    print(prob)
    individuo = (int(BinSearch(prob, prob_, 0, len(prob) - 1)))
    return pop[individuo]

'''
def selectOne(pop, fit):
    fit = fit - min(fit)
    fit = fit / sum(fit)
    max = sum([fitness for fitness in fit])
    print('max')
    print(max)
    pick = random.uniform(0, max)
    current = 0
    for i in range(len(pop)):
        current += fit[i]
        if current > pick:
            print(i)
            return pop[i]
'''


def gac(npop, n_clusters, pcros, pmut, maxit, data_,
        lvmodel, mvmodel, scheme, regression):

    pop = initPopulation(npop, data_, n_clusters)
    bestfit = [0, 0]

    for i in range(0, maxit):
        print("Iteration %s" % (i + 1))
#        fit = [indiv.fitness(data_, n_clusters, lvmodel, mvmodel, scheme,
#                             regression) for indiv in pop]

        fit_ = PyLSboot(len(pop), 8, data_, lvmodel,
                        mvmodel, scheme, regression, 0, 100, nclusters=n_clusters, population=pop)
        fit = fit_.gac()

        new = []
        while len(new) < len(pop):

            # selection
            parent1 = roulettewheel(pop, fit)
            p = uniform(0, 1)

            # genetic operators
            if p <= pcros:
                parent2 = roulettewheel(pop, fit)
#                while parent2 == parent1:
#                    parent2 = roulettewheel(pop, fit)
                child1, child2 = crossover(parent1, parent2, n_clusters)
                new.append(child1)
                if len(new) < len(pop):
                    new.append(child2)
            else:
                child = deepcopy(parent1)
                child.mutation(pmut, n_clusters)
                new.append(child)

        pop = deepcopy(new)

        # elitism
        if max(fit) > bestfit[1]:
            bestfit = [pop[np.argmax(fit)], max(fit)]

    print("\nFitness = %s" % bestfit[1])
    print(bestfit[0].genes)

    output = pd.DataFrame(bestfit[0].genes)
    output.columns = ['Split']
    dataSplit = pd.concat([data_, output], axis=1)

    # return best clusters path matrix
    results = []
    for i in range(n_clusters):
        dataSplited = (dataSplit.loc[dataSplit['Split']
                                     == i]).drop('Split', axis=1)
        dataSplited.index = range(len(dataSplited))
        results.append(PyLSpm(dataSplited, lvmodel, mvmodel, scheme,
                              regression, 0, 100, HOC='true'))

        print(results[i].path_matrix)
        print(results[i].gof())
        print(results[i].residuals()[3])
