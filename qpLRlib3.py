import numpy as np
from gurobipy import *


def otimiza(y, x, size, h, method='fuzzy'):

    n = len(y)
    qsi = 0.0000000001

    model = Model("qQPlr-PLS-PM with fuzzy scheme")
    model.setParam("OutputFlag", 0)
    ac = {}
    awL = {}
    awR = {}

    for i in range(size + 1):
        ac[i] = model.addVar(lb=-GRB.INFINITY, name="ac%d" % i)
        awL[i] = model.addVar(lb=-0.0, name="awL%d" % i)
        awR[i] = model.addVar(lb=-0.0, name="awR%d" % i)

    if method == 'fuzzy':

        k1 = 1  # central
        k2 = 10  # periférica

        if size == 1:
            k2b = 4 / n
        else:
            k2b = 4 / n

    elif method == 'ols':

        k1 = 1  # central
        k2 = 0  # periférica

    y = y.values
    x = x.values

    model.setObjective(k1 * quicksum(((y[i] - ac[0] -
                                       quicksum((ac[j + 1] * x[i, j]) for j in range(size))) *
                                      (y[i] - ac[0] -
                                       quicksum((ac[j + 1] * x[i, j]) for j in range(size))))
                                     for i in range(n))

                       + k2 * quicksum((np.dot(x[:, j], x[:, j].transpose()) *
                                        (awL[j + 1] * awL[j + 1] + awR[j + 1] * awR[j + 1]))
                                       for j in range(size)
                                       ) +

                       + k2 * n * (awL[0] * awL[0] + awR[0] * awR[0]), GRB.MINIMIZE)

    for i in range(0, n):
        model.addConstr(n*ac[0] +
                        quicksum((ac[j + 1] * x[i, j])
                                 for j in range(size))
                        - (1 - h) * (awL[0] + quicksum((abs(x[i, j]) * awL[j + 1])
                                                       for j in range(size))) <= y[i])

    for i in range(0, n):
        model.addConstr(n*ac[0] +
                        quicksum((ac[j + 1] * x[i, j])
                                 for j in range(size))
                        + (1 - h) * (awR[0] + quicksum((abs(x[i, j]) * awR[j + 1])
                                                       for j in range(size))) >= y[i])

    print(awL)
    print(awR)
    model.optimize()
    return [ac[i + 1].x for i in range(size)], [(ac[i + 1].x - awL[i + 1].x) for i in range(size)], [(ac[i + 1].x + awR[i + 1].x) for i in range(size)]


def IC():
    ylow = []
    yhigh = []
    ymid = []
    ylow.append(0)
    ymid.append(0)
    yhigh.append(0)

    for i in range(1, len(y)):
        ylow.append((ac[0].x - awL[0].x) + (ac[1].x - awL[1].x) * x1[i])

    for i in range(1, len(y)):
        yhigh.append((ac[0].x + awR[0].x) + (ac[1].x + awR[1].x) * x1[i])

    for i in range(1, len(y)):
        ymid.append((ylow[i] + yhigh[i]) / 2)

    SST = 0

    for i in range(1, len(y)):
        SST += (y[i] - ylow[i]) * (y[i] - ylow[i]) + \
            (yhigh[i] - y[i]) * (yhigh[i] - y[i])

    SSR = 0

    for i in range(1, len(y)):
        SSR += (ymid[i] - ylow[i]) * (ymid[i] - ylow[i]) + \
            (yhigh[i] - ymid[i]) * (yhigh[i] - ymid[i])

    IC = SSR / SST
    return IC
