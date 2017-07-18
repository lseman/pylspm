# PLSR2
import numpy as np
import pandas as pd
import random


def normaliza(X):
    mean_ = np.mean(X, 0)
    scale_ = np.std(X, 0)
    X = X - mean_
    X = X / (scale_)
    return X

# FOC = preditors (X)
# HOC = response (Y)
# T as scores


def plsr2(X, Y, nc=None, cross='TRUE', seed=None):

    random.seed(seed)

    if (nc == None) and (cross == 'FALSE'):
        nc = 2
    elif (nc == None) and (cross == 'TRUE'):
        nc = X.shape[1]

    X = normaliza(X)
    Y = normaliza(Y)

    X = np.matrix(X)
    Y = np.matrix(Y)

    T = U = W = C = P = Q = B = t = w = u = c = p = q = None

    if cross == 'TRUE':
        tcross = wcross = ucross = ccross = None

        Q2 = np.zeros(shape=(X.shape[1], Y.shape[1]))
        RSS = np.zeros(shape=(X.shape[1], Y.shape[1]))
        PRESS = np.zeros(shape=(X.shape[1], Y.shape[1]))

        sizes = []
        for i in range(10):
            sizes.append(int(len(X) / 10))
        sizes[-1] += (len(X) % 10)

        observation = random.sample(range(len(X)), len(X))

        ini = - np.array(sizes) + np.cumsum(sizes)
        fim = np.cumsum(sizes) - 1

        segments = []
        for i in range(10):
            segments.append(observation[ini[i]:fim[i]])

    epsilon = 6
    maxit = 100
    w_old = 1

    h = 0
    while (h < nc):
        u = Y[:, 0]
        w_old = 1

        for iterations in range(0, maxit):
            w = X.T * u
            w = w / np.dot(u.T, u)
            w = w / np.linalg.norm(w)

            t = X * w

            c = Y.T * t
            c = c / np.dot(t.T, t)

            u = Y * c
            u = u / np.dot(c.T, c)

            w_diff = w - w_old
            w_old = np.copy(w)

            if (np.sum(np.square(w_diff)) < (10**-epsilon)):
                break

        # CROSSVALIDATE
        if cross == 'TRUE':
            RSS[h, :] = sum(np.square(Y - np.dot(t, c.T)))
            press = np.zeros(shape=(10, Y.shape[1]))

            for i in range(10):

                aux = np.ones(len(X), dtype=bool)
                aux[segments[i]] = False

                ucross = Y[aux, 0]
                w_oldcross = 1

                for iterations in range(0, maxit):
                    wcross = X[aux].T * ucross
                    wcross = wcross / np.dot(ucross.T, ucross)
                    wcross = wcross / np.linalg.norm(wcross)

                    tcross = X[aux] * wcross

                    ccross = Y[aux].T * tcross
                    ccross = ccross / np.dot(tcross.T, tcross)

                    ucross = Y[aux] * ccross
                    ucross = ucross / np.dot(ccross.T, ccross)

                    w_diffcross = wcross - w_oldcross
                    w_oldcross = np.copy(wcross)

                    if (np.sum(np.square(w_diffcross)) < (10**-epsilon)):
                        break

                Ycross = np.dot(np.dot(X[~aux], wcross), ccross.T)
                press[i, :] = sum(np.square(Y[~aux] - Ycross))

            PRESS[h, :] = sum(press)
            Q2[h, :] = 1 - (PRESS[h, :] / RSS[h, :])

        # END CROSSVALIDATE

        T = t if T is None else np.hstack((T, t))
        U = u if U is None else np.hstack((U, u))
        W = w if W is None else np.hstack((W, w))
        C = c if C is None else np.hstack((C, c))

        p = X.T * t / np.dot(t.T, t)
        q = Y.T * u / np.dot(u.T, u)

        P = p if P is None else np.hstack((P, p))
        Q = q if Q is None else np.hstack((Q, q))

        X = X - t * p.T
        Y = Y - t * c.T

        if ((cross == 'TRUE') and (sum(Q2[h, :] < 0.0975) == Y.shape[1])):
            break
        h += 1

#    B = W * (P.T * W).I * C.T

    # Scores
    return [T, U]


def HOCcat(data_, mvmodel, seed):
    response = data_.ix[:, 10:25]
    preditors = []
    preditors.append(data_.ix[:, 10:15])
    preditors.append(data_.ix[:, 15:20])
    preditors.append(data_.ix[:, 20:25])

    plsr_ = None
    for i in range(3):
        res_ = plsr2(preditors[i], response, seed=seed)[0]
        plsr_ = res_ if plsr_ is None else np.hstack((plsr_, res_))

    plsr_ = pd.DataFrame(plsr_)
    plsr_.index = range(len(plsr_))

    cols = list(plsr_.columns)
    for s in range(len(cols)):
        cols[cols.index(s)] = 'T' + str(s)
    plsr_.columns = cols

    data_ = pd.concat([data_, plsr_], axis=1)

    Variables = pd.read_csv(mvmodel)
    Variables = Variables[
        Variables.latent.str.contains("Humanização") == False]

    for i in range(len(cols)):
        df_ = pd.DataFrame([['Humanização', cols[i], 'A']],
                           columns=Variables.columns)
        Variables = Variables.append(df_)

    Variables.index = range(len(Variables))
    mvmodel = Variables

    return[data_, mvmodel]
