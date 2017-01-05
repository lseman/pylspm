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

    for i in range(3):
        ac[i] = model.addVar(lb=-GRB.INFINITY, name="ac%d" % i)
        awL[i] = model.addVar(lb=-0.0, name="awL%d" % i)
        awR[i] = model.addVar(lb=-0.0, name="awR%d" % i)

    if method=='fuzzy':
        # para 2
        k1a = 10  # central
        k2a = 1  # periférica

        # para 1
        k1b = 10
        k2b = 1
        k2b2 = 1/n

    if method=='ols':
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
                           k2a * n *(awL[0] * awL[0] + awR[0] * awR[0]), GRB.MINIMIZE)

        for i in range(0, n):
            model.addConstr(ac[0] + x1[i] * ac[1] + x2[i] * ac[2] - (1 - h) * (awL[0] + abs(x1[i]) * awL[1] + abs(x2[i]) * awL[2]) <= y[i])
        for i in range(0, n):
            model.addConstr(ac[0] + x1[i] * ac[1] + x2[i] * ac[2] + (1 - h) * (awR[0] + abs(x1[i]) * awR[1] + abs(x2[i]) * awR[2]) >= y[i])

        model.optimize()
#		print([ac[0].x, awL[0].x, awR[0].x], [ac[1].x, ac[2].x], [awL[1].x, awL[2].x], [awR[1].x, awR[2].x])
        return [ac[1].x, ac[2].x], [ac[1].x - awL[1].x, ac[2].x - awL[2].x], [ac[1].x + awR[1].x, ac[2].x + awR[2].x]

    elif (size == 1):

        x1 = x.ix[:, 0].values
        y = y.values

        model.setObjective(k1b * quicksum(((y[i] - ac[0] - ac[1] * x1[i]) * (y[i] - ac[0] - ac[1] * x1[i])) for i in range(n)) +
                           k2b * ((np.dot(x1, x1.transpose())) * (awL[1] * awL[1] + awR[1] * awR[1])) +
                           k2b2 * n * (awL[0] * awL[0] + awR[0] * awR[0]), GRB.MINIMIZE)

        for i in range(0, n):
            model.addConstr(ac[0] + x1[i] * ac[1] - (1 - h) * (awL[0] + abs(x1[i]) * awL[1]) <= y[i])

        for i in range(0, n):
            model.addConstr(ac[0] + x1[i] * ac[1] + (1 - h) * (awR[0] + abs(x1[i]) * awR[1]) >= y[i])

        model.optimize()
#        print([ac[0].x, awL[0].x, awR[0].x])
        return ac[1].x, ac[1].x - awL[1].x, ac[1].x + awR[1].x
