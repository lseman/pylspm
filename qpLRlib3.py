import numpy as np
from gurobipy import *
import pandas


def otimiza(y, x, size, h, method='fuzzy'):

    n = len(y)
    qsi = 0.0000000000001
#	act = [1, 2, 3]
#	awt = [1, 2, 3]
#	awt1 = [1, 2, 3]
#	awt2 = [1, 2, 3]

    model = Model("qQPlr-PLS-PM with fuzzy scheme")
    model.setParam("OutputFlag", 0)

#	ac = model.addVars(act, lb=-GRB.INFINITY, name="ac")
#	awL = model.addVars(awt1, lb=0.0, name="awL")
#	awR = model.addVars(awt2, lb=0.0, name="awR")

    ac = {}
    awL = {}
    awR = {}

 #   h = 0.5

    for i in range(5):
        ac[i] = model.addVar(lb=-GRB.INFINITY, name="ac%d" % i)
        awL[i] = model.addVar(lb=-0.0, name="awL%d" % i)
        awR[i] = model.addVar(lb=-0.0, name="awR%d" % i)

    if method == 'fuzzy':
        # para 2
        k1a = 1  # central
        k2a = 10  # periférica
        k2a2 = 1 / n

        # para 1
        k1b = 1
        k2b = 10
        k2b2 = 3 / n

    if method == 'ols':
        # para 2
        k1a = 1  # central
        k2a = 0  # periférica

        # para 1
        k1b = 1
        k2b = 0
        k2b2 = 0

    if (size == 2):

        x1 = x.ix[:, 0].values
        x2 = x.ix[:, 1].values
        y = y.values

        model.setObjective(k1a * quicksum(((y[i] - ac[0] - ac[1] * x1[i] - ac[2] * x2[i]) * (y[i] - ac[0] - ac[1] * x1[i] - ac[2] * x2[i])) for i in range(n)) +
                           k2a * ((np.dot(x1, x1.transpose())) * (awL[1] * awL[1] + awR[1] * awR[1])) +
                           k2a * ((np.dot(x2, x2.transpose())) * (awL[2] * awL[2] + awR[2] * awR[2])) +
                           k2a2 * n * (awL[0] * awL[0] + awR[0] * awR[0]), GRB.MINIMIZE)

        for i in range(0, n):
            model.addConstr(ac[0] + x1[i] * ac[1] + x2[i] * ac[2] - (1 - h)
                            * (awL[0] + abs(x1[i]) * awL[1] + abs(x2[i]) * awL[2]) <= y[i])
        for i in range(0, n):
            model.addConstr(ac[0] + x1[i] * ac[1] + x2[i] * ac[2] + (1 - h)
                            * (awR[0] + abs(x1[i]) * awR[1] + abs(x2[i]) * awR[2]) >= y[i])

        model.optimize()
        return [ac[1].x, ac[2].x], [ac[1].x - awL[1].x, ac[2].x - awL[2].x], [ac[1].x + awR[1].x, ac[2].x + awR[2].x]

    elif (size == 1):

        x1 = x.ix[:, 0].values
        y = y.values

        model.setObjective(k1b * quicksum(((y[i] - ac[0] - ac[1] * x1[i]) * (y[i] - ac[0] - ac[1] * x1[i])) for i in range(n)) +
                           k2b * ((np.dot(x1, x1.transpose())) * (awL[1] * awL[1] + awR[1] * awR[1])) +
                           k2b2 * n * (awL[0] * awL[0] + awR[0] * awR[0]), GRB.MINIMIZE)

        for i in range(0, n):
            model.addConstr(ac[0] + x1[i] * ac[1] - (1 - h)
                            * (awL[0] + abs(x1[i]) * awL[1]) <= y[i])

        for i in range(0, n):
            model.addConstr(ac[0] + x1[i] * ac[1] + (1 - h)
                            * (awR[0] + abs(x1[i]) * awR[1]) >= y[i])

        model.optimize()
        return ac[1].x, ac[1].x - awL[1].x, ac[1].x + awR[1].x

    if (size == 3):

        x1 = x.ix[:, 0].values
        x2 = x.ix[:, 1].values
        x3 = x.ix[:, 2].values
        y = y.values

        model.setObjective(k1a * quicksum(((y[i] - ac[0] - ac[1] * x1[i] - ac[2] * x2[i] - ac[3] * x3[i]) * (y[i] - ac[0] - ac[1] * x1[i] - ac[2] * x2[i] - ac[3] * x3[i])) for i in range(n)) +
                           k2a * ((np.dot(x1, x1.transpose())) * (awL[1] * awL[1] + awR[1] * awR[1])) +
                           k2a * ((np.dot(x2, x2.transpose())) * (awL[2] * awL[2] + awR[2] * awR[2])) +
                           k2a * ((np.dot(x3, x3.transpose())) * (awL[3] * awL[3] + awR[3] * awR[3])) +
                           k2a2 * n * (awL[0] * awL[0] + awR[0] * awR[0]), GRB.MINIMIZE)

        for i in range(0, n):
            model.addConstr(ac[0] + x1[i] * ac[1] + x2[i] * ac[2] + x3[i] * ac[3] - (1 - h) * (
                awL[0] + abs(x1[i]) * awL[1] + abs(x2[i]) * awL[2] + abs(x3[i]) * awL[3]) <= y[i])
        for i in range(0, n):
            model.addConstr(ac[0] + x1[i] * ac[1] + x2[i] * ac[2] + x3[i] * ac[3] + (1 - h) * (
                awR[0] + abs(x1[i]) * awR[1] + abs(x2[i]) * awR[2] + abs(x3[i]) * awR[3]) >= y[i])

        model.optimize()
        return [ac[1].x, ac[2].x, ac[3].x], [ac[1].x - awL[1].x, ac[2].x - awL[2].x, ac[3].x - awL[3].x], [ac[1].x + awR[1].x, ac[2].x + awR[2].x, ac[3].x + awR[3].x]

    if (size == 4):

        x1 = x.ix[:, 0].values
        x2 = x.ix[:, 1].values
        x3 = x.ix[:, 2].values
        x4 = x.ix[:, 3].values
        y = y.values

        model.setObjective(k1a * quicksum(((y[i] - ac[0] - ac[1] * x1[i] - ac[2] * x2[i] - ac[3] * x3[i] - ac[4] * x4[i]) *
                                           (y[i] - ac[0] - ac[1] * x1[i] - ac[2] * x2[i] - ac[3] * x3[i] - ac[4] * x4[i])) for i in range(n)) +
                           k2a * ((np.dot(x1, x1.transpose())) * (awL[1] * awL[1] + awR[1] * awR[1])) +
                           k2a * ((np.dot(x2, x2.transpose())) * (awL[2] * awL[2] + awR[2] * awR[2])) +
                           k2a * ((np.dot(x3, x3.transpose())) * (awL[3] * awL[3] + awR[3] * awR[3])) +
                           k2a * ((np.dot(x4, x4.transpose())) * (awL[4] * awL[4] + awR[4] * awR[4])) +
                           k2a2 * n * (awL[0] * awL[0] + awR[0] * awR[0]), GRB.MINIMIZE)

        for i in range(0, n):
            model.addConstr(ac[0] + x1[i] * ac[1] + x2[i] * ac[2] + x3[i] * ac[3] + x4[i] * ac[4] -
                            (1 - h) * (awL[0] + abs(x1[i]) * awL[1] + abs(x2[i]) * awL[2] + abs(x3[i]) * awL[3] + abs(x4[i]) * awL[4]) <= y[i])
        for i in range(0, n):
            model.addConstr(ac[0] + x1[i] * ac[1] + x2[i] * ac[2] + x3[i] * ac[3] + x4[i] * ac[4] +
                            (1 - h) * (awR[0] + abs(x1[i]) * awR[1] + abs(x2[i]) * awR[2] + abs(x3[i]) * awR[3] + abs(x4[i]) * awR[4]) >= y[i])

        model.optimize()
        return [ac[1].x, ac[2].x, ac[3].x, ac[4].x], [ac[1].x - awL[1].x, ac[2].x - awL[2].x, ac[3].x - awL[3].x, ac[4].x - awL[4].x], [ac[1].x + awR[1].x, ac[2].x + awR[2].x, ac[3].x + awR[3].x, ac[4].x + awR[4].x]


def IC():
    ylow = []
    yhigh = []
    ymid = []
    ylow.append(0)
    ymid.append(0)
    yhigh.append(0)

    for i in range(1, len(y)):
        ylow.append((ac[1].x - awL[1].x) + (ac[2].x - awL[2].x) * x1[i])

    for i in range(1, len(y)):
        yhigh.append((ac[1].x + awR[1].x) + (ac[2].x + awR[2].x) * x1[i])

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
