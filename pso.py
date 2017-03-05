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
            particle.velocity = inertia * np.array(particle.velocity) + c1 * np.array(rho1) * np.array(np.array(
                particle.best) - np.array(particle.position)) + c2 * np.array(rho2) * np.array(np.array(bestfit[0].position) - np.array(particle.position))

#            print(particle.velocity)
#            print(np.round(particle.velocity))

            particle.velocity = np.round(particle.velocity)

            particle.position =  np.array(
                particle.position) + np.array(particle.velocity)

            print(particle.position)

            particle.position = np.round(particle.position)

            print(np.round(particle.position))

    fit_ = PyLSboot(len(swarm), 8, data_, lvmodel,
                        mvmodel, scheme, regression, 0, 100, nclusters=n_clusters, population=swarm)
    fit = fit_.pso()
    # best so far
    if max(fit) > bestfit[1]:
        bestfit = [swarm[np.argmax(fit)], max(fit)]
    print("\nFitness = %s" % bestfit[1])

    # return best cluster
    return bestfit[0].position

if __name__ == '__main__':
    print(PSO(int(argv[1]), int(argv[2]), float(argv[3]), float(
        argv[4]), float(argv[5]), float(argv[6]), int(argv[7]), argv[8]))
