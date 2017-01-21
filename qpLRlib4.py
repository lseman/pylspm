import numpy as np
from gurobipy import *

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plota(x, y, ac, awL, awR, xname, yname, size):
    plt.plot(x, y, 'o', markersize=2, label='Dados')
    xvalues = np.arange(min(x), max(x), 0.1)
    ylow = -awL[0].x + [(ac[i] - awL[i + 1].x) *
                        xvalues for i in range(size)][0]
    ymid = [ac[i] * xvalues for i in range(size)][0]
    yhigh = awR[0].x + [(ac[i] + awR[i + 1].x) *
                        xvalues for i in range(size)][0]
    superior = plt.plot(xvalues, ylow, 'b--', label='Limite Inferior')
    centroide = plt.plot(xvalues, ymid, 'k--', label='Centroide')
    inferior = plt.plot(xvalues, yhigh, 'r--', label='Limite Superior')
    plt.legend()
    plt.xlabel(xname[0], fontsize=12)
    plt.ylabel(yname, fontsize=12)
    plt.show()


def otimiza(y, x, size, h, method='fuzzy', plotaIC='false'):

    n = len(y)

    model = Model("qTSQ-PLS-PM with fuzzy scheme")
    model.setParam("OutputFlag", 0)
    awL = {}
    awR = {}

#    h = 0

    for i in range(size + 1):
        awL[i] = model.addVar(lb=-0.0, name="awL%d" % i)
        awR[i] = model.addVar(lb=-0.0, name="awR%d" % i)

    ac, resid = np.linalg.lstsq(x, y)[:2]

    yname = y.name
    xname = x.columns.values
    print(['y: ' + yname])
    print('x: ' + xname)

    y = y.values
    x = x.values

    model.setObjective(quicksum((np.dot(x[:, j], x[:, j].transpose())
                                 * (awL[j + 1] + awR[j + 1]) * (awL[j + 1] + awR[j + 1]))
                                for j in range(size))
                       + (awL[0] + awR[0]) * (awL[0] + awR[0]),
                       GRB.MINIMIZE)

    # Lembrar que no for não vai N

    for i in range(n):
        model.addConstr(
            quicksum((ac[j] * x[i, j])
                     for j in range(size))
            - (1 - h) * (awL[0] + quicksum((abs(x[i, j]) * awL[j + 1])
                                           for j in range(size))) <= y[i])

    for i in range(n):
        model.addConstr(
            quicksum((ac[j] * x[i, j])
                     for j in range(size))
            + (1 - h) * (awR[0] + quicksum((abs(x[i, j]) * awR[j + 1])
                                           for j in range(size))) >= y[i])

    model.optimize()

#    plota(x, y, ac, awL, awR, xname, yname, size)
#    ic = IC(x, y, ac, awL, awR, size)

    if plotaIC == 'false':
        return [ac[i] for i in range(size)], [(ac[i] - awL[i + 1].x) for i in range(size)], [(ac[i] + awR[i + 1].x) for i in range(size)]
    if plotaIC == 'true':
        return model, ac, awL, awR


def IC(x, y, ac, awL, awR, size):
    ylow = []
    yhigh = []
    ymid = []

    for i in range(len(y)):
        ylow.append((0 - awL[0].x) + [(ac[j] - awL[j + 1].x) *
                                      x[i, j] for j in range(size)][0])

    for i in range(len(y)):
        yhigh.append((0 + awR[0].x) + [(ac[j] + awR[j + 1].x) *
                                       x[i, j] for j in range(size)][0])

    for i in range(len(y)):
        ymid.append((ylow[i] + yhigh[i]) / 2)

    SST = 0
    for i in range(len(y)):
        SST += (y[i] - ylow[i]) * (y[i] - ylow[i]) + \
            (yhigh[i] - y[i]) * (yhigh[i] - y[i])

    SSR = 0
    for i in range(len(y)):
        SSR += (ymid[i] - ylow[i]) * (ymid[i] - ylow[i]) + \
            (yhigh[i] - ymid[i]) * (yhigh[i] - ymid[i])

    IC = SSR / SST
    return IC


def plotaIC(y, x, size):
    h = 0
    IClist = []
    hlist = []
    aclist = []

    for i in range(0, 90):
        h += 0.01
        hlist.append(h)
        model, ac, awL, awR = otimiza(y, x, size, h, plotaIC='true')
        aclist.append(awR[1].x)
        IClist.append(IC(x.values, y.values, ac, awL, awR, size))

    x = hlist
    y = IClist
    z = aclist

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('índice h')
    ax.set_ylabel('IC')
    ax.set_zlabel('Coeficiente de Caminho')
    ax.set_axis_bgcolor('white')
    ax.plot(x, y, z)
#    fig.savefig('temp.png', transparent=True)
    plt.show()
