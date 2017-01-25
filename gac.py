from random import randint, uniform
from copy import deepcopy
from sys import argv
import numpy as np
from numpy import inf
import sys
import pandas as pd

from pylspm import PyLSpm

sys.setrecursionlimit(1500)


def isNaN(num):
    return num != num


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


class Individual (object):

    def __init__(self, x, k, genes):
        self.genes = genes
        self.bestelements = 0
        if x is not None:
            for i in range(0, k):
                point = x[randint(0, len(x) - 1)]
                for coord in point:
                    self.genes.append(coord)
            self.dim = len(x[0])
        else:
            self.dim = len(genes) / k

    # assign each point to a cluster

    def assign(self, x):
        output = []
        for point in x:
            distance = []
            for index in range(0, int(len(self.genes) / self.dim)):
                distance.append(np.linalg.norm(np.array(
                    point) - np.array(self.genes[int(index * self.dim):int((index + 1) * self.dim)])))
            output.append(np.argmin(distance))
        self.bestelements = output
        return output

    # windexes of points that belong to a given cluster
    def elements(self, cluster, output):
        return np.where(np.array(output) == cluster)[0]

    # update clusters centroids based on assignments
    def update(self, x, output):
        for index in range(0, int(len(self.genes) / self.dim)):
            xi = self.elements(index, output)
            for d in range(int(index * self.dim), int((index + 1) * self.dim)):
                self.genes[d] = sum([x[item][int(d % self.dim)]
                                     for item in xi]) / len(xi) if len(xi) != 0 else self.genes[d]

    def fitness(self, x):
        output = self.assign(x)
        output = pd.DataFrame(output)
        output.columns = ['Split']
        dataSplit = pd.concat([data_, output], axis=1)

        f1 = []
        results = []
        for i in range(n_clusters):
            dataSplited = (dataSplit.loc[dataSplit['Split']
                                         == i]).drop('Split', axis=1)
            dataSplited.index = range(len(dataSplited))

            if (len(dataSplited) != 0):

                results.append(PyLSpm(dataSplited, lvmodel, mvmodel, scheme,
                                      regression, 0, 50, HOC='true'))

                try:
                    sumOuterResid = pd.DataFrame.sum(
                        pd.DataFrame.sum(results[i].residuals()[1]**2))
                    sumInnerResid = pd.DataFrame.sum(
                        pd.DataFrame.sum(results[i].residuals()[2]**2))
                    f1.append(sumOuterResid + sumInnerResid)
                except :
                    f1.append(10000)
            else:
                f1.append(10000)

        self.update(x, output)

        return (1/np.sum(f1))

    def mutation(self, pmut):
        for g, gene in enumerate(self.genes):
            if uniform(0, 1) <= pmut:
                delta = uniform(0, 1)
                if uniform(0, 1) <= 0.5:
                    self.genes[g] = gene - 2 * delta * \
                        gene if gene != 0 else -2 * delta
                else:
                    self.genes[g] = gene + 2 * delta * \
                        gene if gene != 0 else 2 * delta

    def getPoints(self):
        return self.bestelements


def GAPopulationInit(npop, x, k):
    return [Individual(x, k, []) for i in range(0, npop)]


def Crossover(parent1, parent2, k):
    print('fez crossover')
    point = randint(1, len(parent1.genes) - 2)
    return Individual(None, k, parent1.genes[:point] + parent2.genes[point:]), Individual(None, k, parent2.genes[:point] + parent1.genes[point:])


def RouletteWheel(pop, fit):
    sumf = sum(fit)
    prob = [(item + sum(fit[:index])) / sumf for index, item in enumerate(fit)]
    return pop[int(BinSearch(prob, uniform(0, 1), 0, len(prob) - 1))]


def GeneticAlg(npop, k, pcros, pmut, maxit, dados):
    x = dados
    pop = GAPopulationInit(npop, x, k)
    fit = [indiv.fitness(x) for indiv in pop]
    verybest = [pop[np.argmax(fit)], max(fit)]
    for i in range(0, maxit):
        print("Iteration %s" % (i + 1))
        fit = [indiv.fitness(x) for indiv in pop]
        new = []
        while len(new) < len(pop):
            # selection
            parent1 = RouletteWheel(pop, fit)
            p = uniform(0, 1)
            # genetic operators
            if p <= pcros:
                parent2 = RouletteWheel(pop, fit)
                while parent2 == parent1:
                    parent2 = RouletteWheel(pop, fit)
                child1, child2 = Crossover(parent1, parent2, k)
                new.append(child1)
                if len(new) < len(pop):
                    new.append(child2)
            else:
                child = deepcopy(parent1)
                child.mutation(pmut)
                new.append(child)
        pop = deepcopy(new)
        # elitism (but individual is kept outside population)
        if max(fit) > verybest[1]:
            verybest = [pop[np.argmax(fit)], max(fit)]
            output = pop[np.argmax(fit)].getPoints()
    print("\nFitness = %s" % verybest[1])
    # return best cluster
    output = pd.DataFrame(verybest[0].getPoints())
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

#    return verybest[0].genes

n_individuals = 10
n_clusters = 2
p_crossover = 0.6
p_mutation = 0.1
iterations = 10

data = 'dados_miss.csv'
lvmodel = 'lvnew.csv'
mvmodel = 'mvnew.csv'
scheme = 'path'
regression = 'ols'
data_ = pd.read_csv(data)

data_ = pd.read_csv(data)
mean = pd.DataFrame.mean(data_)
for j in range(len(data_.columns)):
    for i in range(len(data_)):
        if (isNaN(data_.ix[i, j])):
            data_.ix[i, j] = mean[j]

segmenta = GeneticAlg(n_individuals, n_clusters,
                      p_crossover, p_mutation, iterations, data_.values)
