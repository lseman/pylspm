# Function that deletes two edges and reverses the sequence in between the
# deleted edges
import random
import math

from random import randint, uniform
from copy import deepcopy
import numpy as np
from numpy import inf
import pandas as pd
import random
import heapq

from pylspm import PyLSpm
from boot import PyLSboot

from multiprocessing import Pool, freeze_support


class TabuSearch(object):

    def stochasticTwoOpt(self, perm):
        result = perm[:]  # make a copy
        size = len(result)
        # select indices of two random points in the tour
        p1, p2 = random.randrange(0, size), random.randrange(0, size)
        # do this so as not to overshoot tour boundaries
        exclude = set([p1])
        if p1 == 0:
            exclude.add(size - 1)
        else:
            exclude.add(p1 - 1)

        if p1 == size - 1:
            exclude.add(0)
        else:
            exclude.add(p1 + 1)

        while p2 in exclude:
            p2 = random.randrange(0, size)

        # to ensure we always have p1<p2
        if p2 < p1:
            p1, p2 = p2, p1

        # now reverse the tour segment between p1 and p2
        result[p1:p2] = reversed(result[p1:p2])

        return result

    def locateBestCandidate(self, candidates):
        candidates.sort(key=lambda c: c["candidate"]["cost"])
        best = candidates[0]["candidate"]
        return best

    def fit(self, node, data, nclusters, lvmodel, mvmodel, scheme, regression, goal=1000):
        output = pd.DataFrame(node)
        output.columns = ['Split']
        dataSplit = pd.concat([data, output], axis=1)
        f1 = []
        results = []
        for i in range(nclusters):
            dataSplited = (dataSplit.loc[dataSplit['Split']
                                         == i]).drop('Split', axis=1)
            dataSplited.index = range(len(dataSplited))

            try:
                results.append(PyLSpm(dataSplited, lvmodel, mvmodel,
                                      scheme, regression, 0, 50, HOC='true'))

                resid = results[i].residuals()[3]
                f1.append(resid)

            except:
                f1.append(10000)

        cost = (np.sum(f1))
        print(1 / cost)
        return cost

    def generateCandidates(self, item):

        check_tabu = None
        result = {}
        permutation = self.stochasticTwoOpt(self.best["permutation"])

        while check_tabu == None:
            if permutation in self.tabuList:
                print('deu igual')
                permutation = self.stochasticTwoOpt(self.best["permutation"])
            else:
                check_tabu = 1

        candidate = {}
        candidate["permutation"] = permutation
        candidate["cost"] = self.fit(candidate["permutation"], self.data_,
                                self.n_clusters, self.lvmodel, self.mvmodel, self.scheme, self.regression)
        result["candidate"] = candidate

        return result

    def __init__(self, tabu_size, n_children, n_clusters, n_goal, iterations, data_,
             lvmodel, mvmodel, scheme, regression):

        self.data_ = data_
        self.lvmodel = lvmodel
        self.mvmodel = mvmodel
        self.scheme = scheme
        self.n_clusters = n_clusters
        self.n_children = n_children
        self.tabu_size = tabu_size
        self.regression = regression

        node = []
        for i in range(len(data_)):
            node.append(random.randrange(n_clusters))


        self.best = {}
        self.best["permutation"] = node
        self.best["cost"] = self.fit(self.best["permutation"], self.data_, self.n_clusters,
                           self.lvmodel, self.mvmodel, self.scheme, self.regression)

        print(self.best)

        self.tabuList = []

        for i in range(0, iterations):

            print("Iteration %s" % (i + 1))

            candidates = []

            p = Pool(8)
            candidates = p.map(self.generateCandidates, range(n_children))
            p.close()
            p.join()

            bestCandidate = self.locateBestCandidate(candidates)

            if bestCandidate["cost"] < self.best["cost"]:
                self.best = bestCandidate
                self.tabuList.append(bestCandidate["permutation"])
                if len(self.tabuList) > tabu_size:
                    print('apagou 1')
                    del self.tabuList[0]

            iterations -= 1

        return best
