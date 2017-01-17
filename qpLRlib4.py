import numpy as np
from gurobipy import *


def otimiza(y, x, size, h, method='fuzzy'):

    n = len(y)

    model = Model("qTSQ-PLS-PM with fuzzy scheme")
    model.setParam("OutputFlag", 0)
    awL = {}
    awR = {}

    h=0

    for i in range(size + 1):
        awL[i] = model.addVar(lb=-0.0, name="awL%d" % i)
        awR[i] = model.addVar(lb=-0.0, name="awR%d" % i)

    ac, resid = np.linalg.lstsq(x, y)[:2]

    y = y.values
    x = x.values

    model.setObjective(quicksum((np.dot(x[:, j], x[:, j].transpose())
                                 * (awL[j + 1] + awR[j + 1]) * (awL[j + 1] + awR[j + 1]))
                                for j in range(size))
                       + (awL[0] + awR[0]) * (awL[0] + awR[0]),
                       GRB.MINIMIZE)

    for i in range(0, n):
        model.addConstr(
            quicksum((ac[j] * x[i, j])
                     for j in range(size))
            - (1 - h) * (awL[0] + quicksum((abs(x[i, j]) * awL[j + 1])
                                           for j in range(size))) <= y[i])

    for i in range(0, n):
        model.addConstr(
            quicksum((ac[j] * x[i, j])
                     for j in range(size))
            + (1 - h) * (awR[0] + quicksum((abs(x[i, j]) * awR[j + 1])
                                           for j in range(size))) >= y[i])

    model.optimize()
    print(awL)
    print(awR)
    return [ac[i] for i in range(size)], [(ac[i] - awL[i + 1].x) for i in range(size)], [(ac[i] + awR[i + 1].x) for i in range(size)]


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
