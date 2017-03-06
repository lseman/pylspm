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


class Particle (object):

    def __init__(self, data_, n_clusters):
        self.position = []
        if not self.position:
            for i in range(len(data_)):
                self.position.append(random.randrange(n_clusters))
        print(self.position)
        self.velocity = [0 for cluster in self.position]
        self.best = deepcopy(self.position)
        self.bestfit = 0


def PSOPopulationInit(npart, x, n_clusters):
    return [Particle(x, n_clusters) for i in range(0, npart)]


def pso(npart, n_clusters, in_max, in_min, c1, c2, maxit,  data_,
        lvmodel, mvmodel, scheme, regression):

    swarm = PSOPopulationInit(npart, data_, n_clusters)

    bestfit = [0, 0]

    rho1 = [uniform(0, 1) for i in range(0, len(data_))]
    rho2 = [uniform(0, 1) for i in range(0, len(data_))]

    for i in range(0, maxit):
        inertia = (in_max - in_min) * ((maxit - i + 1) / maxit) + in_min
        print("Iteration %s" % (i + 1))
        fit_ = PyLSboot(len(swarm), 8, data_, lvmodel,
                        mvmodel, scheme, regression, 0, 100, nclusters=n_clusters, population=swarm)
        fit = fit_.pso()

        bestfit = [swarm[np.argmax(fit)], max(fit)]

        # update best
        for index, particle in enumerate(swarm):
            if fit[index] > particle.bestfit:
                particle.best = deepcopy(particle.position)
                particle.bestfit = deepcopy(fit[index])

        # update velocity and position
        for particle in swarm:
            particle.velocity = 0*inertia * np.array(particle.velocity) + c1 * np.array(rho1) * np.array(np.array(
                particle.best) - np.array(particle.position)) + c2 * np.array(rho2) * np.array(np.array(bestfit[0].position) - np.array(particle.position))

#            print(particle.velocity)

            particle.position += particle.velocity

            for j in range(len(particle.position)):
                if particle.position[j] <= 0:
                    particle.position[j] = 0
                elif particle.position[j] > (n_clusters-1):
                    particle.position[j] = (n_clusters-1)

            particle.position = np.round(particle.position)
#            print('corrigido')
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
        print(results[i].residuals()[3])