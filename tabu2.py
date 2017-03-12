from random import randint, uniform
from copy import deepcopy
import numpy as np
from numpy import inf
import pandas as pd
import random

from pylspm import PyLSpm
from boot import PyLSboot

from multiprocessing import Pool, freeze_support


def stochasticTwoOpt(perm):
    result = perm[:]  # make a copy
    size = len(result)
    p1, p2 = random.randrange(0, size), random.randrange(0, size)
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

    if p2 < p1:
        p1, p2 = p2, p1

    result[p1:p2] = reversed(result[p1:p2])

    return result


def locateBestCandidate(candidates):
    candidates.sort(key=lambda c: c[1])
    best = candidates[0]
    return best


def generateCandidates(best, tabuList):

    check_tabu = None
    permutation = stochasticTwoOpt(best[0])

    while check_tabu == None:
        if permutation in tabuList:
            permutation = stochasticTwoOpt(best[0])
        else:
            check_tabu = 1

    return permutation


def tabu(tabu_size, n_children, n_clusters, n_goal, iterations, data_,
         lvmodel, mvmodel, scheme, regression):

    node = []
    for i in range(len(data_)):
        node.append(random.randrange(n_clusters))

    best_ = PyLSboot(1, 8, data_, lvmodel,
                     mvmodel, scheme, regression, 0, 100, nclusters=n_clusters, population=[node])
    best = best_.tabu()[0]
    tabuList = []

    for i in range(0, iterations):

        print("Iteration %s" % (i + 1))
        candidates = []

        for index in range(0, n_children):
            candidates.append(generateCandidates(best, tabuList))

        fit_ = PyLSboot(len(candidates), 8, data_, lvmodel,
                        mvmodel, scheme, regression, 0, 100, nclusters=n_clusters, population=candidates)
        fit = fit_.tabu()

        bestCandidate = locateBestCandidate(fit)

        if bestCandidate[1] < best[1]:
            best = [bestCandidate[0], bestCandidate[1]]
            tabuList.append(bestCandidate[0])
            if len(tabuList) > tabu_size:
                print('apagou 1')
                del tabuList[0]

    print("\nFitness = %s" % best[1])
    print(bestfit[0])

    output = pd.DataFrame(best[0])
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
