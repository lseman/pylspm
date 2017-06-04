# PyLS-PM Library
# Author: Laio Oriel Seman
# Creation: November 2016
# Description: Library based on Juan Manuel Velasquez Estrada's simplePLS,
# Gaston Sanchez's plspm and Mikko Rönkkö's matrixpls made in R

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
from qpLRlib4 import otimiza, plotaIC
import scipy.linalg
from collections import Counter
from pca import *
from pandas.plotting import scatter_matrix


class PyLSpm(object):

    def BTS(self):

        n = self.data_.shape[0]
        p = self.data_.shape[1]

        chi2 = -(n - 1 - (2 * p + 5) / 6) * \
            np.log(np.linalg.det(pd.DataFrame.corr(self.data_)))
        df = p * (p - 1) / 2

        pvalue = scipy.stats.distributions.chi2.sf(chi2, df)

        return [chi2, pvalue]

    def KMO(self):

        cor_ = pd.DataFrame.corr(self.data_)
        invCor = np.linalg.inv(cor_)
        rows = cor_.shape[0]
        cols = cor_.shape[1]
        A = np.ones((rows, cols))

        for i in range(rows):
            for j in range(i, cols):
                A[i, j] = - (invCor[i, j]) / \
                    (np.sqrt(invCor[i, i] * invCor[j, j]))
                A[j, i] = A[i, j]

        num = np.sum(np.sum((cor_)**2)) - np.sum(np.sum(np.diag(cor_**2)))
        den = num + (np.sum(np.sum(A**2)) - np.sum(np.sum(np.diag(A**2))))
        kmo = num / den

        return kmo

    def PCA(self):
        for i in range(self.lenlatent):
            print(self.latent[i])
            block = self.data_[self.Variables['measurement']
                               [self.Variables['latent'] == self.latent[i]]]
            PCAdo(block, self.latent[i])

    def scatterMatrix(self):
        for i in range(1, self.lenlatent):
            block = self.data[self.Variables['measurement']
                              [self.Variables['latent'] == self.latent[i]]]
            scatter_matrix(block, diagonal='kde')

            plt.savefig('imgs/scatter' + self.latent[i], bbox_inches='tight')
            plt.clf()
            plt.cla()

    def sampleSize(self):
        r = 0.3
        alpha = 0.05
#        power=0.9

        C = 0.5 * np.log((1 + r) / (1 - r))

        Za = scipy.stats.norm.ppf(1 - (0.05 / 2))

        sizeArray = []
        powerArray = []

        power = 0.5
        for i in range(50, 100, 1):
            power = i / 100
            powerArray.append(power)

            Zb = scipy.stats.norm.ppf(1 - power)
            N = abs((Za - Zb) / C)**2 + 3

            sizeArray.append(N)

        return [powerArray, sizeArray]

    def normaliza(self, X):
        correction = np.sqrt((len(X) - 1) / len(X))  # std factor corretion
        mean_ = np.mean(X, 0)
        scale_ = np.std(X, 0)
        X = X - mean_
        X = X / (scale_ * correction)
        return X

    def gof(self):
        r2mean = np.mean(self.r2.T[self.endoexo()[0]].values)
        AVEmean = self.AVE().copy()

        totalblock = 0
        for i in range(self.lenlatent):
            block = self.data_[self.Variables['measurement']
                               [self.Variables['latent'] == self.latent[i]]]
            block = len(block.columns.values)
            totalblock += block
            AVEmean[self.latent[i]] = AVEmean[self.latent[i]] * block

        AVEmean = np.sum(AVEmean) / totalblock
        return np.sqrt(AVEmean * r2mean)

    def endoexo(self):
        exoVar = []
        endoVar = []

        for i in range(self.lenlatent):
            if(self.latent[i] in self.LVariables['target'].values):
                endoVar.append(self.latent[i])
            else:
                exoVar.append(self.latent[i])

        return endoVar, exoVar

    def residuals(self):
        exoVar = []
        endoVar = []

        outer_residuals = self.data.copy()
#        comun_ = self.data.copy()

        for i in range(self.lenlatent):
            if(self.latent[i] in self.LVariables['target'].values):
                endoVar.append(self.latent[i])
            else:
                exoVar.append(self.latent[i])

        for i in range(self.lenlatent):
            block = self.data_[self.Variables['measurement']
                               [self.Variables['latent'] == self.latent[i]]]
            block = block.columns.values

            loadings = self.outer_loadings.ix[
                block][self.latent[i]].values

            outer_ = self.fscores.ix[:, i].values
            outer_ = outer_.reshape(len(outer_), 1)
            loadings = loadings.reshape(len(loadings), 1)
            outer_ = np.dot(outer_, loadings.T)

            outer_residuals.ix[:, block] = self.data_.ix[
                :, block] - outer_
#            comun_.ix[:, block] = outer_

        inner_residuals = self.fscores[endoVar]
        inner_ = pd.DataFrame.dot(self.fscores, self.path_matrix.ix[endoVar].T)
        inner_residuals = self.fscores[endoVar] - inner_

        residuals = pd.concat([outer_residuals, inner_residuals], axis=1)

        mean_ = np.mean(self.data, 0)

#        comun_ = comun_.apply(lambda row: row + mean_, axis=1)

        sumOuterResid = pd.DataFrame.sum(
            pd.DataFrame.sum(outer_residuals**2))
        sumInnerResid = pd.DataFrame.sum(
            pd.DataFrame.sum(inner_residuals**2))

        divFun = sumOuterResid + sumInnerResid

        return residuals, outer_residuals, inner_residuals, divFun

    def srmr(self):
        srmr = (self.empirical() - self.implied())
        srmr = np.sqrt(((srmr.values) ** 2).mean())
        return srmr

    def implied(self):
        corLVs = pd.DataFrame.cov(self.fscores)
        implied_ = pd.DataFrame.dot(self.outer_loadings, corLVs)
        implied = pd.DataFrame.dot(implied_, self.outer_loadings.T)

        implied.values[[np.arange(len(self.manifests))] * 2] = 1
        return implied

    def empirical(self):
        empirical = self.data_
        return pd.DataFrame.corr(empirical)

    def frequency(self, data=None, manifests=None):

        if data is None:
            data = self.data

        if manifests is None:
            manifests = self.manifests

        frequencia = pd.DataFrame(0, index=range(1, 6), columns=manifests)

        for i in range(len(manifests)):
            frequencia[manifests[i]] = data[
                manifests[i]].value_counts()

        frequencia = frequencia / len(data) * 100
        frequencia = frequencia.reindex_axis(
            sorted(frequencia.columns), axis=1)
        frequencia = frequencia.fillna(0).T
        frequencia = frequencia[(frequencia.T != 0).any()]

        maximo = pd.DataFrame.max(pd.DataFrame.max(data, axis=0))

        if int(maximo) & 1:
            neg = np.sum(frequencia.ix[:, 1: ((maximo - 1) / 2)], axis=1)
            ind = frequencia.ix[:, ((maximo + 1) / 2)]
            pos = np.sum(
                frequencia.ix[:, (((maximo + 1) / 2) + 1):maximo], axis=1)

        else:
            neg = np.sum(frequencia.ix[:, 1:((maximo) / 2)], axis=1)
            ind = 0
            pos = np.sum(frequencia.ix[:, (((maximo) / 2) + 1):maximo], axis=1)

        frequencia['Neg.'] = pd.Series(
            neg, index=frequencia.index)
        frequencia['Ind.'] = pd.Series(
            ind, index=frequencia.index)
        frequencia['Pos.'] = pd.Series(
            pos, index=frequencia.index)

        return frequencia

    def frequencyPlot(self, data_, SEM=None):

        segmento = 'SEM'
        SEMmax = pd.DataFrame.max(SEM)

        ok = None

        for i in range(1, self.lenlatent):
            block = data_[self.Variables['measurement']
                          [self.Variables['latent'] == self.latent[i]]]
            block = pd.concat([block, SEM], axis=1)

            for j in range(SEMmax + 1):
                dataSEM = (block.loc[data_[segmento] == j]
                           ).drop(segmento, axis=1)
                block_val = dataSEM.columns.values
                dataSEM = self.frequency(dataSEM, block_val)['Pos.']
                dataSEM = dataSEM.rename(j + 1)
                ok = dataSEM if ok is None else pd.concat(
                    [ok, dataSEM], axis=1)

        for i in range(1, self.lenlatent):
            block = data_[self.Variables['measurement']
                          [self.Variables['latent'] == self.latent[i]]]

            block_val = block.columns.values
            plotando = ok.ix[block_val].dropna(axis=1)
            plotando.plot.bar()
            plt.legend(loc='upper center',
                           bbox_to_anchor=(0.5, -.08), ncol=6)
            plt.savefig('imgs/frequency' + self.latent[i], bbox_inches='tight')
            plt.clf()
            plt.cla()
#            plt.show()

#                block.plot.bar()
#                plt.show()

        '''for i in range(1, self.lenlatent):
            block = self.data[self.Variables['measurement']
                              [self.Variables['latent'] == self.latent[i]]]

            block_val = block.columns.values
            block = self.frequency(block, block_val)

            block.plot.bar()
            plt.show()'''

    def dataInfo(self):
        sd_ = np.std(self.data, 0)
        mean_ = np.mean(self.data, 0)
        skew = scipy.stats.skew(self.data)
        kurtosis = scipy.stats.kurtosis(self.data)
        w = [scipy.stats.shapiro(self.data.ix[:, i])[0]
             for i in range(len(self.data.columns))]

        return [mean_, sd_, skew, kurtosis, w]

    def predict(self, method='redundancy'):
        exoVar = []
        endoVar = []

        for i in range(self.lenlatent):
            if(self.latent[i] in self.LVariables['target'].values):
                endoVar.append(self.latent[i])
            else:
                exoVar.append(self.latent[i])

        if (method == 'exogenous'):
            Beta = self.path_matrix.ix[endoVar][endoVar]
            Gamma = self.path_matrix.ix[endoVar][exoVar]

            beta = [1 if (self.latent[i] in exoVar)
                    else 0 for i in range(self.lenlatent)]

            beta = np.diag(beta)

            beta_ = [1 for i in range(len(Beta))]
            beta_ = np.diag(beta_)

            beta = pd.DataFrame(beta, index=self.latent, columns=self.latent)

            mid = pd.DataFrame.dot(Gamma.T, np.linalg.inv(beta_ - Beta.T))
            mid = (mid.T.values).flatten('F')

            k = 0
            for j in range(len(exoVar)):
                for i in range(len(endoVar)):
                    beta.ix[endoVar[i], exoVar[j]] = mid[k]
                    k += 1

        elif (method == 'redundancy'):
            beta = self.path_matrix.copy()
            beta_ = pd.DataFrame(1, index=np.arange(
                len(exoVar)), columns=np.arange(len(exoVar)))
            beta.ix[exoVar, exoVar] = np.diag(np.diag(beta_.values))

        elif (method == 'communality'):
            beta = np.diag(np.ones(len(self.path_matrix)))
            beta = pd.DataFrame(beta)

        partial_ = pd.DataFrame.dot(self.outer_weights, beta.T.values)
        prediction = pd.DataFrame.dot(partial_, self.outer_loadings.T.values)
        predicted = pd.DataFrame.dot(self.data, prediction)
        predicted.columns = self.manifests

        mean_ = np.mean(self.data, 0)
        intercept = mean_ - np.dot(mean_, prediction)

        predictedData = predicted.apply(lambda row: row + intercept, axis=1)

        return predictedData

    def cr(self):
        # Composite Reliability
        composite = pd.DataFrame(0, index=np.arange(1), columns=self.latent)

        for i in range(self.lenlatent):
            block = self.data_[self.Variables['measurement']
                               [self.Variables['latent'] == self.latent[i]]]
            p = len(block.columns)

            if(p != 1):
                cor_mat = np.cov(block.T)
                evals, evecs = np.linalg.eig(cor_mat)
                U, S, V = np.linalg.svd(cor_mat, full_matrices=False)

                indices = np.argsort(evals)
                indices = indices[::-1]
                evecs = evecs[:, indices]
                evals = evals[indices]

                loadings = V[0, :] * np.sqrt(evals[0])

                numerador = np.sum(abs(loadings))**2
                denominador = numerador + (p - np.sum(loadings ** 2))
                cr = numerador / denominador
                composite[self.latent[i]] = cr

            else:
                composite[self.latent[i]] = 1

        composite = composite.T
        return(composite)

    def r2adjusted(self):
        n = len(self.data_)
        r2 = self.r2.values

        r2adjusted = pd.DataFrame(0, index=np.arange(1), columns=self.latent)

        for i in range(self.lenlatent):
            p = sum(self.LVariables['target'] == self.latent[i])
            r2adjusted[self.latent[i]] = r2[i] - \
                (p * (1 - r2[i])) / (n - p - 1)

        return r2adjusted.T

    def htmt(self):

        htmt_ = pd.DataFrame(pd.DataFrame.corr(self.data_),
                             index=self.manifests, columns=self.manifests)

        mean = []
        allBlocks = []
        for i in range(self.lenlatent):
            block_ = self.Variables['measurement'][
                self.Variables['latent'] == self.latent[i]]
            allBlocks.append(list(block_.values))
            block = htmt_.ix[block_, block_]
            mean_ = (block - np.diag(np.diag(block))).values
            mean_[mean_ == 0] = np.nan
            mean.append(np.nanmean(mean_))

        comb = [[k, j] for k in range(self.lenlatent)
                for j in range(self.lenlatent)]

        comb_ = [(np.sqrt(mean[comb[i][1]] * mean[comb[i][0]]))
                 for i in range(self.lenlatent ** 2)]

        comb__ = []
        for i in range(self.lenlatent ** 2):
            block = (htmt_.ix[allBlocks[comb[i][1]],
                              allBlocks[comb[i][0]]]).values
#            block[block == 1] = np.nan
            comb__.append(np.nanmean(block))

        htmt__ = np.divide(comb__, comb_)
        where_are_NaNs = np.isnan(htmt__)
        htmt__[where_are_NaNs] = 0

        htmt = pd.DataFrame(np.tril(htmt__.reshape(
            (self.lenlatent, self.lenlatent)), k=-1), index=self.latent, columns=self.latent)

        return htmt

    def comunalidades(self):
        # Comunalidades
        return self.outer_loadings**2

    def AVE(self):
        # AVE
        return self.comunalidades().apply(lambda column: column.sum() / (column != 0).sum())

    def rhoA(self):
        # rhoA
        rhoA = pd.DataFrame(0, index=np.arange(1), columns=self.latent)

        for i in range(self.lenlatent):
            weights = pd.DataFrame(self.outer_weights[self.latent[i]])
            weights = weights[(weights.T != 0).any()]
            result = pd.DataFrame.dot(weights.T, weights)
            result_ = pd.DataFrame.dot(weights, weights.T)

            S = self.data_[self.Variables['measurement'][
                self.Variables['latent'] == self.latent[i]]]
            S = pd.DataFrame.dot(S.T, S) / S.shape[0]
            numerador = (
                np.dot(np.dot(weights.T, (S - np.diag(np.diag(S)))), weights))
            denominador = (
                (np.dot(np.dot(weights.T, (result_ - np.diag(np.diag(result_)))), weights)))
            rhoA_ = ((result)**2) * (numerador / denominador)
            if(np.isnan(rhoA_.values)):
                rhoA[self.latent[i]] = 1
            else:
                rhoA[self.latent[i]] = rhoA_.values

        return rhoA.T

    def xloads(self):
        # Xloadings
        A = self.data_.transpose().values
        B = self.fscores.transpose().values
        A_mA = A - A.mean(1)[:, None]
        B_mB = B - B.mean(1)[:, None]

        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)

        xloads_ = (np.dot(A_mA, B_mB.T) /
                   np.sqrt(np.dot(ssA[:, None], ssB[None])))
        xloads = pd.DataFrame(
            xloads_, index=self.manifests, columns=self.latent)

        return xloads

    def corLVs(self):
        # Correlations LVs
        corLVs_ = np.tril(pd.DataFrame.corr(self.fscores))
        return pd.DataFrame(corLVs_, index=self.latent, columns=self.latent)

    def alpha(self):
        # Cronbach Alpha
        alpha = pd.DataFrame(0, index=np.arange(1), columns=self.latent)

        for i in range(self.lenlatent):
            block = self.data_[self.Variables['measurement']
                               [self.Variables['latent'] == self.latent[i]]]
            p = len(block.columns)

            if(p != 1):
                p_ = len(block)
                correction = np.sqrt((p_ - 1) / p_)
                soma = np.var(np.sum(block, axis=1))
                cor_ = pd.DataFrame.corr(block)

                denominador = soma * correction**2
                numerador = 2 * np.sum(np.tril(cor_) - np.diag(np.diag(cor_)))

                alpha_ = (numerador / denominador) * (p / (p - 1))
                alpha[self.latent[i]] = alpha_
            else:
                alpha[self.latent[i]] = 1

        return alpha.T

    def vif(self):
        vif = []
        totalmanifests = range(len(self.data_.columns))
        for i in range(len(totalmanifests)):
            independent = [x for j, x in enumerate(totalmanifests) if j != i]
            coef, resid = np.linalg.lstsq(
                self.data_.ix[:, independent], self.data_.ix[:, i])[:2]

            r2 = 1 - resid / \
                (self.data_.ix[:, i].size * self.data_.ix[:, i].var())

            vif.append(1 / (1 - r2))

        vif = pd.DataFrame(vif, index=self.manifests)
        return vif

    def PLSc(self):
        ##################################################
        # PLSc

        rA = self.rhoA()
        corFalse = self.corLVs()

        for i in range(self.lenlatent):
            for j in range(self.lenlatent):
                if i == j:
                    corFalse.ix[i][j] = 1
                else:
                    corFalse.ix[i][j] = corFalse.ix[i][
                        j] / np.sqrt(rA.ix[self.latent[i]] * rA.ix[self.latent[j]])

        corTrue = np.zeros([self.lenlatent, self.lenlatent])
        for i in range(self.lenlatent):
            for j in range(self.lenlatent):
                corTrue[j][i] = corFalse.ix[i][j]
                corTrue[i][j] = corFalse.ix[i][j]

        corTrue = pd.DataFrame(corTrue, corFalse.columns, corFalse.index)

        # Loadings

        attenuedOuter_loadings = pd.DataFrame(
            0, index=self.manifests, columns=self.latent)

        for i in range(self.lenlatent):
            weights = pd.DataFrame(self.outer_weights[self.latent[i]])
            weights = weights[(weights.T != 0).any()]
            result = pd.DataFrame.dot(weights.T, weights)
            result_ = pd.DataFrame.dot(weights, weights.T)

            newLoad = (
                weights.values * np.sqrt(rA.ix[self.latent[i]].values)) / (result.values)
            myindex = self.Variables['measurement'][
                self.Variables['latent'] == self.latent[i]]
            myindex_ = self.latent[i]
            attenuedOuter_loadings.ix[myindex.values, myindex_] = newLoad

        # Path

        dependent = np.unique(self.LVariables.ix[:, 'target'])

        for i in range(len(dependent)):
            independent = self.LVariables[self.LVariables.ix[
                :, "target"] == dependent[i]]["source"]

            dependent_ = corTrue.ix[dependent[i], independent]
            independent_ = corTrue.ix[independent, independent]

#            path = np.dot(np.linalg.inv(independent_),dependent_)
            coef, resid = np.linalg.lstsq(independent_, dependent_)[:2]
            self.path_matrix.ix[dependent[i], independent] = coef

        return attenuedOuter_loadings

        # End PLSc
        ##################################################

    def __init__(self, dados, LVcsv, Mcsv, scheme='path', regression='ols', h=0, maximo=300, stopCrit=7, HOC='false', disattenuate='false',
                 method='lohmoller'):
        self.data = dados
        self.LVcsv = LVcsv
        self.Mcsv = Mcsv
        self.maximo = maximo
        self.stopCriterion = stopCrit
        self.h = h
        self.scheme = scheme
        self.regression = regression
        self.disattenuate = disattenuate

        contador = 0
        convergiu = 0

        data = dados if type(
            dados) is pd.core.frame.DataFrame else pd.read_csv(dados)

        LVariables = pd.read_csv(LVcsv)

        Variables = Mcsv if type(
            Mcsv) is pd.core.frame.DataFrame else pd.read_csv(Mcsv)

        latent_ = LVariables.values.flatten('F')
        latent__ = np.unique(latent_, return_index=True)[1]
#        latent = np.unique(latent_)
        latent = [latent_[i] for i in sorted(latent__)]

        self.lenlatent = len(latent)

        # Repeating indicators

        if (HOC == 'true'):

            data_temp = pd.DataFrame()

            for i in range(self.lenlatent):
                block = self.data[Variables['measurement']
                                  [Variables['latent'] == latent[i]]]
                block = block.columns.values
                data_temp = pd.concat(
                    [data_temp, data[block]], axis=1)

            cols = list(data_temp.columns)
            counts = Counter(cols)
            for s, num in counts.items():
                if num > 1:
                    for suffix in range(1, num + 1):
                        cols[cols.index(s)] = s + '.' + str(suffix)
            data_temp.columns = cols

            doublemanifests = list(Variables['measurement'].values)
            counts = Counter(doublemanifests)
            for s, num in counts.items():
                if num > 1:
                    for suffix in range(1, num + 1):
                        doublemanifests[doublemanifests.index(
                            s)] = s + '.' + str(suffix)

            Variables['measurement'] = doublemanifests
            data = data_temp

        # End data manipulation

        manifests_ = Variables['measurement'].values.flatten('F')
        manifests__ = np.unique(manifests_, return_index=True)[1]
        manifests = [manifests_[i] for i in sorted(manifests__)]

        self.manifests = manifests
        self.latent = latent
        self.Variables = Variables
        self.LVariables = LVariables

        data = data[manifests]
        data_ = self.normaliza(data)
        self.data = data
        self.data_ = data_

        outer_weights = pd.DataFrame(0, index=manifests, columns=latent)
        for i in range(len(Variables)):
            outer_weights[Variables['latent'][i]][
                Variables['measurement'][i]] = 1

        inner_paths = pd.DataFrame(0, index=latent, columns=latent)

        for i in range(len(LVariables)):
            inner_paths[LVariables['source'][i]][LVariables['target'][i]] = 1

        path_matrix = inner_paths.copy()

        if method == 'wold':
            fscores = pd.DataFrame.dot(data_, outer_weights)
            intera = self.lenlatent
            intera_ = 1

        # LOOP
        for iterations in range(0, self.maximo):
            contador = contador + 1

            if method == 'lohmoller':
                fscores = pd.DataFrame.dot(data_, outer_weights)
                intera = 1
                intera_ = self.lenlatent
#               fscores = self.normaliza(fscores) # Old Mode A

            for q in range(intera):

                # Schemes
                if (scheme == 'path'):
                    for h in range(intera_):

                        i = h if method == 'lohmoller' else q

                        follow = (path_matrix.ix[i, :] == 1)
                        if (sum(follow) > 0):
                            # i ~ follow
                            inner_paths.ix[inner_paths[follow].index, i] = np.linalg.lstsq(
                                fscores.ix[:, follow], fscores.ix[:, i])[0]

                        predec = (path_matrix.ix[:, i] == 1)
                        if (sum(predec) > 0):
                            semi = fscores.ix[:, predec]
                            a_ = list(fscores.ix[:, i])

                            cor = [sp.stats.pearsonr(a_, list(semi.ix[:, j].values.flatten()))[
                                0] for j in range(len(semi.columns))]
                            inner_paths.ix[inner_paths[predec].index, i] = cor

                elif (scheme == 'fuzzy'):
                    for i in range(len(path_matrix)):
                        follow = (path_matrix.ix[i, :] == 1)
                        if (sum(follow) > 0):
                            ac, awL, awR = otimiza(fscores.ix[:, i], fscores.ix[
                                                   :, follow], len(fscores.ix[:, follow].columns), 0)
                            inner_paths.ix[inner_paths[follow].index, i] = ac

                        predec = (path_matrix.ix[:, i] == 1)
                        if (sum(predec) > 0):
                            cor, p = sp.stats.pearsonr(list(fscores.ix[:, i]), list(
                                fscores.ix[:, predec].values.flatten()))
                            inner_paths.ix[inner_paths[predec].index, i] = cor

                elif (scheme == 'centroid'):
                    inner_paths = np.sign(pd.DataFrame.multiply(
                        pd.DataFrame.corr(fscores), (path_matrix + path_matrix.T)))

                elif (scheme == 'factor'):
                    inner_paths = pd.DataFrame.multiply(
                        pd.DataFrame.corr(fscores), (path_matrix + path_matrix.T))

                if method == 'wold':
                    fscores[self.latent[q]] = pd.DataFrame.dot(
                        fscores, inner_paths)
                elif method == 'lohmoller':
                    fscores = pd.DataFrame.dot(fscores, inner_paths)

            last_outer_weights = outer_weights.copy()

            # Outer Weights
            for i in range(self.lenlatent):

                # Reflexivo / Modo A
                if(Variables['mode'][Variables['latent'] == latent[i]]).any() == "A":
                    a = data_[Variables['measurement'][
                        Variables['latent'] == latent[i]]]
                    b = fscores.ix[:, latent[i]]

                    # 1/N (Z dot X)
                    res_ = (1 / len(data_)) * np.dot(b, a)

                    myindex = Variables['measurement'][
                        Variables['latent'] == latent[i]]
                    myindex_ = latent[i]
                    outer_weights.ix[myindex.values,
                                     myindex_] = res_ / np.linalg.norm(res_)  # New Mode A

                # Formativo / Modo B
                elif(Variables['mode'][Variables['latent'] == latent[i]]).any() == "B":

                    a = data_[Variables['measurement'][
                        Variables['latent'] == latent[i]]]

                    # (X'X)^-1 X'Y
                    a_ = np.dot(a.T, a)
                    inv_ = np.linalg.inv(a_)
                    res_ = np.dot(np.dot(inv_, a.T),
                                  fscores.ix[:, latent[i]])

                    myindex = Variables['measurement'][
                        Variables['latent'] == latent[i]]
                    myindex_ = latent[i]
                    outer_weights.ix[myindex.values,
                                     myindex_] = res_ / np.norm(outer_weights, 0)

            if method == 'wold':
                fscores = pd.DataFrame.dot(fscores, inner_paths)

            # New Mode A
#            outer_weights = outer_weights / \
#                (np.std(outer_weights, 0)) #* len(fscores)**2)

#                outer_weights = outer_weights / np.std(outer_weights, 0)
#                print(outer_weights)

            diff_ = ((abs(last_outer_weights) -
                      abs(outer_weights))**2).values.sum()

            if (diff_ < (10**(-(self.stopCriterion)))):
                convergiu = 1
                break
            # END LOOP

        print(contador)

        # Bootstraping trick
        if(np.isnan(outer_weights).any().any()):
            return None

        # Standardize Outer Weights (w / || scores ||)
        divide_ = np.diag(1 / (np.std(np.dot(data_, outer_weights), 0)
                               * np.sqrt((len(data_) - 1) / len(data_))))
        outer_weights = np.dot(outer_weights, divide_)
        outer_weights = pd.DataFrame(
            outer_weights, index=manifests, columns=latent)

        fscores = pd.DataFrame.dot(data_, outer_weights)

        # Outer Loadings

        outer_loadings = pd.DataFrame(0, index=manifests, columns=latent)

        for i in range(self.lenlatent):
            a = data_[Variables['measurement'][
                Variables['latent'] == latent[i]]]
            b = fscores.ix[:, latent[i]]

            cor_ = [sp.stats.pearsonr(a.ix[:, j], b)[0]
                    for j in range(len(a.columns))]

            myindex = Variables['measurement'][
                Variables['latent'] == latent[i]]
            myindex_ = latent[i]
            outer_loadings.ix[myindex.values, myindex_] = cor_

        # Paths

        if (regression == 'fuzzy'):
            path_matrix_low = path_matrix.copy()
            path_matrix_high = path_matrix.copy()
            path_matrix_range = path_matrix.copy()

        r2 = pd.DataFrame(0, index=np.arange(1), columns=latent)
        dependent = np.unique(LVariables.ix[:, 'target'])

        for i in range(len(dependent)):
            independent = LVariables[LVariables.ix[
                :, "target"] == dependent[i]]["source"]
            dependent_ = fscores.ix[:, dependent[i]]
            independent_ = fscores.ix[:, independent]

            if (self.regression == 'ols'):
                # Path Normal
                coef, resid = np.linalg.lstsq(independent_, dependent_)[:2]
#                model = sm.OLS(dependent_, independent_)
#                results = model.fit()
#                print(results.summary())
#                r2[dependent[i]] = results.rsquared

                r2[dependent[i]] = 1 - resid / \
                    (dependent_.size * dependent_.var())

                path_matrix.ix[dependent[i], independent] = coef
#                pvalues.ix[dependent[i], independent] = results.pvalues

            elif (self.regression == 'fuzzy'):
                size = len(independent_.columns)
                ac, awL, awR = otimiza(dependent_, independent_, size, self.h)

#                plotaIC(dependent_, independent_, size)

                ac, awL, awR = (ac[0], awL[0], awR[0]) if (
                    size == 1) else (ac, awL, awR)

                path_matrix.ix[dependent[i], independent] = ac
                path_matrix_low.ix[dependent[i], independent] = awL
                path_matrix_high.ix[dependent[i], independent] = awR

                # Matrix Fuzzy

                for i in range(len(path_matrix.columns)):
                    for j in range(len(path_matrix.columns)):
                        path_matrix_range.ix[i, j] = str(round(
                            path_matrix_low.ix[i, j], 3)) + ' ; ' + str(round(path_matrix_high.ix[i, j], 3))

        r2 = r2.T

        self.path_matrix = path_matrix
        self.outer_weights = outer_weights
        self.fscores = fscores

        #################################
        # PLSc
        if disattenuate == 'true':
            outer_loadings = self.PLSc()
        ##################################

        # Path Effects

        indirect_effects = pd.DataFrame(0, index=latent, columns=latent)

        path_effects = [None] * self.lenlatent
        path_effects[0] = self.path_matrix

        for i in range(1, self.lenlatent):
            path_effects[i] = pd.DataFrame.dot(
                path_effects[i - 1], self.path_matrix)
        for i in range(1, len(path_effects)):
            indirect_effects = indirect_effects + path_effects[i]

        total_effects = indirect_effects + self.path_matrix

        if (regression == 'fuzzy'):
            self.path_matrix_high = path_matrix_high
            self.path_matrix_low = path_matrix_low
            self.path_matrix_range = path_matrix_range

        self.total_effects = total_effects.T
        self.indirect_effects = indirect_effects
        self.outer_loadings = outer_loadings
        self.contador = contador
        self.convergiu = convergiu

        self.r2 = r2

    def impa(self):

        # Unstandardized Scores

        scale_ = np.std(self.data, 0)
        outer_weights_ = pd.DataFrame.divide(
            self.outer_weights, scale_, axis=0)
        relativo = pd.DataFrame.sum(outer_weights_, axis=0)

        for i in range(len(outer_weights_)):
            for j in range(len(outer_weights_.columns)):
                outer_weights_.ix[i, j] = (
                    outer_weights_.ix[i, j]) / relativo[j]

        unstandardizedScores = pd.DataFrame.dot(self.data, outer_weights_)

        # Rescaled Scores

        rescaledScores = pd.DataFrame(0, index=range(
            len(self.data)), columns=self.latent)

        for i in range(self.lenlatent):
            block = self.data[self.Variables['measurement'][
                self.Variables['latent'] == self.latent[i]]]

            maximo = pd.DataFrame.max(block, axis=0)
            minimo = pd.DataFrame.min(block, axis=0)
            minimo_ = pd.DataFrame.min(minimo)
            maximo_ = pd.DataFrame.max(maximo)

            rescaledScores[self.latent[
                i]] = 100 * (unstandardizedScores[self.latent[i]] - minimo_) / (maximo_ - minimo_)

        # Manifests Indirect Effects

        manifestsIndEffects = pd.DataFrame(
            self.outer_weights, index=self.manifests, columns=self.latent)

        effect_ = pd.DataFrame(
            self.outer_weights, index=self.manifests, columns=self.latent)

        for i in range(len(self.latent[i])):
            effect_ = pd.DataFrame.dot(effect_, self.path_matrix.T)
            manifestsIndEffects = manifestsIndEffects + effect_

        # Peformance Scores LV

        performanceScoresLV = pd.DataFrame.mean(rescaledScores, axis=0)

        # Performance Manifests

        maximo = pd.DataFrame.max(self.data, axis=0)
        minimo = pd.DataFrame.min(self.data, axis=0)
        diff = maximo - minimo
        performanceManifests = pd.DataFrame.subtract(
            self.data, minimo.T, axis=1)
        performanceManifests = pd.DataFrame.divide(
            performanceManifests, diff, axis=1)
        performanceManifests = performanceManifests * 100

        performanceManifests = pd.DataFrame.mean(performanceManifests, axis=0)

        # Unstandardized Path

        unstandardizedPath = pd.DataFrame(
            0, index=self.latent, columns=self.latent)

        dependent = np.unique(self.LVariables.ix[:, 'target'])
        for i in range(len(dependent)):
            independent = self.LVariables[self.LVariables.ix[
                :, "target"] == dependent[i]]["source"]
            dependent_ = unstandardizedScores.ix[:, dependent[i]]
            independent_ = unstandardizedScores.ix[:, independent]

#            ac, awL, awR = otimiza(dependent_, independent_, len(
#                independent_.columns), self.h, 'ols')
            A = np.vstack(
                [independent_.T.values, np.ones(len(independent_))]).T
            coef, resid = np.linalg.lstsq(A, dependent_)[:2]

            unstandardizedPath.ix[dependent[i], independent] = coef[:-1]

        # Unstandardized Total Effects

        path_effects = [None] * self.lenlatent
        path_effects[0] = unstandardizedPath

        unstandardizedIndirectEffects = pd.DataFrame(
            0, index=self.latent, columns=self.latent)

        for i in range(1, self.lenlatent):
            path_effects[i] = pd.DataFrame.dot(
                path_effects[i - 1], unstandardizedPath)
        for i in range(1, len(path_effects)):
            unstandardizedIndirectEffects = unstandardizedIndirectEffects + \
                path_effects[i]

        unstandardizedPathTotal = unstandardizedIndirectEffects + unstandardizedPath

        # Unstandardized Manifests Indirect Effects

        unstandardizedManifestsIndEffects = pd.DataFrame(
            outer_weights_, index=self.manifests, columns=self.latent)

        unstandardizedEffect_ = pd.DataFrame(
            outer_weights_, index=self.manifests, columns=self.latent)

        for i in range(len(self.latent[i])):
            unstandardizedEffect_ = pd.DataFrame.dot(
                unstandardizedEffect_, unstandardizedPath.T)
            unstandardizedManifestsIndEffects = unstandardizedManifestsIndEffects + \
                unstandardizedEffect_

        # IMPAplot

        for i in range(self.lenlatent):
            for j in range(len(performanceManifests)):
                # if unstandardizedManifestsIndEffects.ix[j,i] != 0:
                plt.plot(unstandardizedManifestsIndEffects.ix[
                         j, i], performanceManifests[j], 'o', label=self.manifests[j])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.12), ncol=6)
            plt.xlim(0)
            plt.ylim(0, 100)
            plt.xlabel('Efeito Total')
            plt.ylabel(self.latent[i])
            plt.savefig('imgs/impaManifests' +
                        self.latent[i], bbox_inches='tight')
            plt.clf()
            plt.cla()
#            plt.show()

        for i in range(self.lenlatent):
            if (np.sum(unstandardizedPathTotal.ix[i, :] != 0) > 1):
                for j in range(self.lenlatent):
                    if unstandardizedPathTotal.ix[i, j] != 0:
                        plt.plot(unstandardizedPathTotal.ix[
                            i, j], performanceScoresLV.ix[j], 'o', label=self.latent[j])
                plt.xlabel('Efeito Total')
                plt.ylabel(self.latent[i])
                plt.legend(loc='upper center',
                           bbox_to_anchor=(0.5, -.12), ncol=6)
                plt.xlim(0, 1)
                plt.ylim(0, 100)
                plt.savefig('imgs/impaScores' +
                            self.latent[i], bbox_inches='tight')
                plt.clf()
                plt.cla()
#                plt.show()

        return [performanceScoresLV, performanceManifests, unstandardizedPathTotal, unstandardizedManifestsIndEffects]
