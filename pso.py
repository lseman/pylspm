# PLS-PM particle swarm clustering
# Author: Laio Oriel Seman

# JARBOUI, B. et al. Combinatorial particle swarm optimization (CPSO) for
# partitional clustering problem. Applied Mathematics and Computation, v.
# 192, n. 2, p. 337–345, set. 2007.

# Based on https://github.com/liviaalmeida/clustering

from random import randint, uniform
from copy import deepcopy
import numpy as np
from numpy import inf
import pandas as pd
import random

from pylspm import PyLSpm
from boot import PyLSboot
from call_mpi import PyLSmpi


class Particle(object):

    def __init__(self, data_, n_clusters):
        self.position = []
        if not self.position:
            for i in range(len(data_)):
                self.position.append(random.randrange(n_clusters))
        print(self.position)

        self.velocity = [0 for clusterPoint in self.position]
        self.S = [0 for clusterPoint in self.position]
        self.best = deepcopy(self.position)
        self.bestfit = 0


def PSOSwarmInit(npart, x, n_clusters):
    return [Particle(x, n_clusters) for i in range(0, npart)]


def sigmoid(x, M):
    return M / (1 + np.exp(-x))


def pso(cores, npart, n_clusters, in_max, in_min, c1, c2, maxit,  data_,
        lvmodel, mvmodel, scheme, regression):

    swarm = PSOSwarmInit(npart, data_, n_clusters)

    rho1 = [uniform(0, 1) for i in range(0, len(data_))]
    rho2 = [uniform(0, 1) for i in range(0, len(data_))]

    bestfit = [0, 0]

    alfa = 0.35

    equalOpt = [-1, 1]

    for i in range(0, maxit):

        print("Iteration %s" % (i + 1))

        inertia = (in_max - in_min) * ((maxit - i + 1) / maxit) + in_min

        fit = PyLSmpi('pso', len(swarm), cores, data_, lvmodel,
                      mvmodel, scheme, regression, 0, 100, nclusters=n_clusters, population=swarm)
#        fit_ = PyLSboot(len(swarm), 8, data_, lvmodel,
#                        mvmodel, scheme, regression, 0, 100, nclusters=n_clusters, population=swarm)
#        fit = fit_.pso()

        if max(fit) > bestfit[1]:
            bestfit = [swarm[np.argmax(fit)], max(fit)]

        # update best
        for index, particle in enumerate(swarm):
            if fit[index] > particle.bestfit:
                particle.best = deepcopy(particle.position)
                particle.bestfit = deepcopy(fit[index])

        # update velocity and position
        for particle in swarm:

            for i in range(len(particle.position)):
                if (particle.position[i] == bestfit[0].position[i]) and (particle.position[i] == particle.best[i]):
                    y = equalOpt[random.randint(0, 1)]
                elif particle.position[i] == bestfit[0].position[i]:
                    y = 1
                elif particle.position[i] == particle.best[i]:
                    y = -1
                else:
                    y = 0

                particle.velocity[i] = particle.velocity[
                    i] + (c1 * rho1[i] * (-1 - y)) + (c2 * rho2[i] * (1 - y))
                lmbd = particle.velocity[i] + y

                if lmbd > alfa:
                    ynew = 1
                elif lmbd < -alfa:
                    ynew = -1
                else:
                    ynew = 0

                if ynew == 1:
                    particle.position[i] = bestfit[0].position[i]
                elif ynew == -1:
                    particle.position[i] = particle.best[i]
                else:
                    oldpos = particle.position[i]
                    while particle.position[i] == oldpos:
                        particle.position[i] = random.randint(
                            0, n_clusters - 1)

#            particle.S = sigmoid(particle.velocity, n_clusters)

#            for j in range(len(particle.position)):
#                particle.position[j] = np.round(
# particle.S[j] + (n_clusters - 1) * sigma * np.random.randn())

#            particle.position = np.round(
#                particle.S + (n_clusters - 1) * sigma * np.random.randn())

            """particle.position = particle.position + particle.velocity

            for j in range(len(particle.position)):
                if particle.position[j] > n_clusters - 1:
                    particle.position[j] = n_clusters - 1
                elif particle.position[j] <= 0:
                    particle.position[j] = 0

            particle.position = np.round(particle.position)"""
#            print(particle.position)

    fit_ = PyLSboot(len(swarm), 8, data_, lvmodel,
                    mvmodel, scheme, regression, 0, 100, nclusters=n_clusters, population=swarm)
    fit = fit_.pso()

    # best so far
    if max(fit) > bestfit[1]:
        bestfit = [swarm[np.argmax(fit)], max(fit)]
    print("\nFitness = %s" % bestfit[1])

    # return best cluster
    print(bestfit[0].position)

    output = pd.DataFrame(bestfit[0].position)
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
