# PyLS-PM Library
# Author: Laio Oriel Seman
# Creation: November 2016
# Description: Library based on Juan Manuel Velasquez Estrada's simplePLS, Gaston Sanchez's plspm and Mikko Rönkkö's matrixpls made in R
# This library implements a new internal path scheme, entitled Fuzzy Scheme
# and calculates Path Coefficients using a Quadratic Possibilistic Regression

import pandas as pd
import numpy as np
import copy
import scipy as sp
import scipy.stats
import csv
from qpLRlib3 import otimiza

class PyLSpm(object):

    def normaliza(self, X):
        mean_ = np.mean(X, 0)
        scale_ = np.std(X, 0)
        X = X - mean_
        X = X / scale_
        return X

    def desnormaliza(self, X):
        mean_ = np.mean(X, 0)
        scale_ = np.std(X, 0)

        X = X * scale_
        X = X + mean_
        return X

    def runIMPA(self):

        # Unstandardized Scores
        scale_ = np.std(self.data, 0)
        outer_weights_ = pd.DataFrame.divide (self.outer_weights, scale_, axis=0)
        relativo = pd.DataFrame.sum (outer_weights_, axis=0)

        for i in range(len(outer_weights_)):
            for j in range(len(outer_weights_.columns)):
                outer_weights_.ix[i,j] = (outer_weights_.ix[i,j])/relativo[j]

        unstandardizedScores = pd.DataFrame.dot(self.data,outer_weights_)

        # Rescaled Scores
        rescaledScores = pd.DataFrame(0, index=range(len(self.data)), columns=self.latent)

        for i in range(len(self.latent)):
            block = self.data[self.Variables ['measurement'][self.Variables['latent']==self.latent[i]]]

            maximo = pd.DataFrame.max(block, axis=0)
            minimo = pd.DataFrame.min(block, axis=0)
            minimo_ = pd.DataFrame.min(minimo)
            maximo_ = pd.DataFrame.max(maximo)

            rescaledScores[self.latent[i]] = 100*(unstandardizedScores[self.latent[i]]-minimo_)/(maximo_-minimo_)

        # Manifests Indirect Effects

        manifestsIndEffects = pd.DataFrame(self.outer_weights, index=self.manifests, columns=self.latent)

        effect_ = pd.DataFrame(self.outer_weights, index=self.manifests, columns=self.latent)

        for i in range(len(self.latent[i])):
            effect_ = pd.DataFrame.dot(effect_, self.path_matrix.T)
            manifestsIndEffects = manifestsIndEffects + effect_

        # Peformance Scores LV

        performanceScoresLV = pd.DataFrame.mean(rescaledScores, axis=0)

        # Performance Manifests

        maximo = pd.DataFrame.max(self.data, axis=0)
        minimo = pd.DataFrame.min(self.data, axis=0)
        diff = maximo-minimo
        performanceManifests = pd.DataFrame.subtract (self.data, minimo.T, axis=1)
        performanceManifests = pd.DataFrame.divide (performanceManifests, diff, axis = 1)
        performanceManifests = performanceManifests * 100

        # Unstandardized Path

        unstandardizedPath = pd.DataFrame(0, index=self.latent, columns=self.latent)

        dependant=np.unique(self.LVariables.ix[:,'target'])
        for i in range(len(dependant)):
            independant=self.LVariables[self.LVariables.ix[:,"target"]==dependant[i]]["source"]
            dependant_ = unstandardizedScores.ix[:,dependant[i]]
            independant_ = unstandardizedScores.ix[:,independant]

            coef, resid = np.linalg.lstsq(independant_, dependant_)[:2]
            unstandardizedPath.ix[dependant[i],independant] = coef

        print(self.path_matrix)
        print(unstandardizedPath)

        return [unstandardizedScores, performanceScoresLV, manifestsIndEffects]

    def runPrediction(self, method='exogenous'):
        exoVar = []
        endoVar = []

        for i in range(len(self.latent)):
            if(self.latent[i] in self.LVariables['target'].values):
                endoVar.append(self.latent[i])
            else:
                exoVar.append(self.latent[i])

        if (method=='exogenous'):
            # Exogenous
            Beta = self.path_matrix.ix[endoVar][endoVar]
            Gamma = self.path_matrix.ix[endoVar][exoVar]

            beta = []
            for i in range(len(self.latent)):
                if (self.latent[i] in exoVar):
                    beta.append(1)
                else:
                    beta.append(0)

            beta=np.diag(beta)
      
            beta_ = []
            for i in range(len(Beta)):
                beta_.append(1)
            beta_=np.diag(beta_)

            mid = pd.DataFrame.dot(Gamma.T,np.linalg.inv(beta_-Beta.T))

            beta = pd.DataFrame(beta, index=self.latent, columns=self.latent)
            mid=(mid.T.values).flatten('F')

            k=0
            for j in range(len(exoVar)):
                for i in range(len(endoVar)):
                    beta.ix[endoVar[i],exoVar[j]] = mid[k]
                    k+=1

        if (method=='redundancy'):
        # Redundancy
            beta = self.path_matrix
            beta_ = pd.DataFrame(1, index=np.arange(len(exoVar)), columns=np.arange(len(exoVar)))
            beta.ix[exoVar,exoVar] = np.diag(np.diag(beta_.values))

        partial_ = pd.DataFrame.dot(self.outer_weights, beta.T.values)
        prediction = pd.DataFrame.dot(partial_, self.outer_loadings.T.values)
        predicted = pd.DataFrame.dot(self.data, prediction)
        predicted = pd.DataFrame(predicted)
        predicted.columns = self.manifests

        mean_ = np.mean(self.data, 0)
        intercept = mean_ - np.dot(mean_,prediction)

        predictedData = predicted.apply(lambda row: row + intercept, axis=1)

        residuals = predictedData - self.data

        return predictedData

    def runSRMR(self):
        srmr = (pd.DataFrame.corr(self.data_) - self.runImplied())
        srmr = np.sqrt(((srmr.values) ** 2).mean())
        return srmr

    def runImplied(self):
        corLVs = pd.DataFrame.cov(self.fscores)
        implied_ = pd.DataFrame.dot(self.outer_loadings,corLVs)
        implied = pd.DataFrame.dot(implied_, self.outer_loadings.T)
        implied.values[[np.arange(len(self.manifests))]*2] = 1
        return implied

    def runEmpirical(self):
        return pd.DataFrame.corr(self.data)

    def runCR(self):
        # Composite Reliability
        composite = pd.DataFrame(0, index=np.arange(1), columns=self.latent)

        for i in range(len(self.latent)):
            block = self.data_[self.Variables ['measurement'][self.Variables['latent']==self.latent[i]]]
            p = len(block.columns)

            cor_mat = np.cov(block.T)
            evals, evecs = np.linalg.eig(cor_mat)
            U, S, V = np.linalg.svd(cor_mat, full_matrices=False)

            indices = np.argsort(evals)
            indices = indices[::-1]
            evecs = evecs[:,indices]
            evals = evals[indices]

            loadings = V[0,:]*np.sqrt(evals[0])

            numerador = np.sum(abs(loadings))**2
            denominador = numerador + (p - np.sum(loadings*loadings))
            cr = numerador/denominador
            composite[self.latent[i]] = cr

        composite = composite.T
        return(composite)

    def runHTMT(self):

        htmt_ = pd.DataFrame.corr(self.data)
        htmt_ = pd.DataFrame(htmt_, index=htmt_.index, columns= htmt_.index)

        mean = []
        allBlocks = []
        for i in range(len(self.latent)):
            block_ = self.Variables ['measurement'][self.Variables['latent']==self.latent[i]]
            allBlocks.append(list(block_.values))
            block = htmt_.ix[block_, block_]
            mean_ = (block - np.diag(np.diag(block))).values
            mean_[mean_ == 0] = np.nan
            mean.append(np.nanmean(mean_))

        comb = []
        for j in range(len(self.latent)):
            for k in range(len(self.latent)):
                comb.append([k, j])

        comb_ = []
        for i in range(len(self.latent)*len(self.latent)):
            comb_.append(np.sqrt(mean[comb[i][1]]*mean[comb[i][0]]))

        comb__ = []
        comb__2 = []
        for i in range(len(self.latent)*len(self.latent)):
            block = (htmt_.ix[allBlocks[comb[i][1]], allBlocks[comb[i][0]]]).values
            block[block == 1] = np.nan
            comb__.append(np.nanmean(block))

        htmt__ = np.divide(comb__,comb_)
        htmt = pd.DataFrame(np.tril(htmt__.reshape((5, 5))), index=self.latent, columns=self.latent)

        return htmt

    def runComunalidades(self):
        # Comunalidades
        return self.outer_loadings**2

    def runAVE(self):
        # AVE
        return self.runComunalidades().apply(lambda column: column.sum()/(column != 0).sum())

    def runRhoA(self):
        # rhoA
        rhoA = pd.DataFrame(0, index=np.arange(1), columns=self.latent)

        for i in range(len(self.latent)):
            weights = pd.DataFrame(self.outer_weights[self.latent[i]])
            weights = weights[(weights.T != 0).any()]
            result=pd.DataFrame.dot(weights.T,weights)
            result_=pd.DataFrame.dot(weights,weights.T)

            S = self.data_[self.Variables ['measurement'][self.Variables['latent']==self.latent[i]]]
            S = pd.DataFrame.dot(S.T, S) / S.shape[0]
            numerador = (np.dot(np.dot(weights.T,(S-np.diag(np.diag(S)))),weights))
            denominador = ((np.dot(np.dot(weights.T,(result_-np.diag(np.diag(result_)))),weights)))
            rhoA_=((result)**2)*(numerador/denominador)
            rhoA[self.latent[i]]=rhoA_.values

        return rhoA.T

    def runXloads(self):
        # Xloadings
        A = self.data_.transpose().values
        B = self.fscores.transpose().values
        A_mA = A - A.mean(1)[:,None]
        B_mB = B - B.mean(1)[:,None]

        ssA = (A_mA**2).sum(1);
        ssB = (B_mB**2).sum(1);

        xloads_ = (np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None])))
        xloads = pd.DataFrame(xloads_, index=self.manifests, columns=self.latent)

        return xloads

    def runCorLVs(self):
        # Correlations LVs
        corLVs_ = np.tril(pd.DataFrame.corr(self.fscores))
        return pd.DataFrame(corLVs_, index=self.latent, columns=self.latent)

    def runAlphas(self):
        # Cronbach Alpha
        alpha = pd.DataFrame(0, index=np.arange(1), columns=self.latent)

        for i in range(len(self.latent)):
            block = self.data_[self.Variables ['measurement'][self.Variables['latent']==self.latent[i]]]
            p = len(block.columns)
            p_ = len(block)
            correction = np.sqrt((p_-1)/p_)
            soma = np.var(np.sum(block, axis=1))
            cor_ =pd.DataFrame.corr(block)

            denominador = soma * correction**2
            numerador = 2 * np.sum(np.tril(cor_)-np.diag(np.diag(cor_)))

            alpha_ = (numerador / denominador) * (p / (p - 1))
            alpha[self.latent[i]]=alpha_

        return alpha.T

    def __init__(self, dados, LVcsv, Mcsv, scheme='path', regression='ols', h=0, maximo=300, stopCrit=7):
        self.data = dados
        self.LVcsv = LVcsv
        self.Mcsv = Mcsv
        self.maximo = maximo
        self.stopCriterion = stopCrit
        self.h = h
        self.scheme = scheme
        self.regression = regression

        contador = 0

        if type(dados) is pd.core.frame.DataFrame:
            data = dados
        else:
            data = pd.read_csv(dados)
            
        LVariables = pd.read_csv(LVcsv)
        Variables = pd.read_csv(Mcsv)
        manifests = list(data.columns.values)
        latent = list(np.unique(LVariables))

        self.manifests = manifests
        self.latent = latent
        self.Variables = Variables
        self.LVariables = LVariables

        data_ = self.normaliza(data)
        self.data = data
        self.data_ = data_

        outer_weights = pd.DataFrame(0, index=manifests, columns=latent)

        for i in range(len(Variables)):
            outer_weights[Variables['latent'][i]] [Variables['measurement'][i]] = 1

        inner_paths = pd.DataFrame(0, index=latent, columns=latent)

        indirect_effects = copy.deepcopy(inner_paths)

        for i in range(len(LVariables)):
            inner_paths[LVariables['source'][i]] [LVariables['target'][i]] = 1

        path_matrix = copy.deepcopy(inner_paths)

        # LOOP
        for iterations in range(0,self.maximo):
            contador = contador + 1
            fscores = pd.DataFrame.dot(data_,outer_weights)
            fscores = self.normaliza(fscores)

            # Schemes
            if (scheme=='path'):
                for i in range(len(path_matrix)):
                    follow = (path_matrix.iloc[i,:]) == 1
                    if (sum(follow) > 0):
                        # i ~ follow
                        inner_paths.ix[inner_paths[follow].index,i]=(np.linalg.lstsq(fscores.loc[:,follow], fscores.iloc[:,i])[0])

                    predec = (path_matrix.iloc[:,i] == 1)
                    if (sum(predec) > 0):
                        cor, p = sp.stats.pearsonr(list(fscores.iloc[:,i]), list(fscores.ix[:,predec].values.flatten()))
                        inner_paths.ix[inner_paths[predec].index,i] = cor

            elif (scheme=='fuzzy'):
                for i in range(len(path_matrix)):
                    follow = (path_matrix.iloc[i,:]) == 1
                    if (sum(follow) > 0):
                        ac, awL, awR = otimiza(fscores.iloc[:,i], fscores.loc[:,follow], len(fscores.loc[:,follow].columns), 0)
                        inner_paths.ix[inner_paths[follow].index,i]= ac

                    predec = (path_matrix.iloc[:,i] == 1)
                    if (sum(predec) > 0):
                        cor, p = sp.stats.pearsonr(list(fscores.iloc[:,i]), list(fscores.ix[:,predec].values.flatten()))
                        inner_paths.ix[inner_paths[predec].index,i] = cor

            elif (scheme=='centroid'):
                inner_paths = np.sign(pd.DataFrame.dot ( pd.DataFrame.corr(fscores), (path_matrix+path_matrix.T)))

            elif (scheme=='factor'):
                inner_paths = (pd.DataFrame.multiply ( pd.DataFrame.corr(fscores), (path_matrix+path_matrix.T)))

            fscores = pd.DataFrame.dot(fscores,inner_paths)
            fscores = self.normaliza(fscores)
            last_outer_weights = copy.deepcopy(outer_weights)

            # Outer Weights
            for i in range(len(latent)):
                  
                # Formativo / Modo B
                if(Variables ['mode'][Variables['latent']==latent[i]]).any()=="B":

                    a = data_[Variables ['measurement'][Variables['latent']==latent[i]]]
                    a__ = []
                    for j in range(len(a.columns)):
                        a__.append(list(a.ix[:,j]))
                    lin_ = np.corrcoef(a__)
                    inv_ = (np.linalg.inv(lin_))

                    cor_ = []
                    for j in range(len(a.columns)):
                        b = list(fscores.ix[:,latent[i]])
                        a_ = list(a.ix[:,j])
                        cor, p = sp.stats.pearsonr(a_, b)
                        cor_.append(cor)
 
                    myindex = Variables ['measurement'][Variables['latent']==latent[i]]
                    myindex_= latent[i]            
                    outer_weights.ix[myindex.values, myindex_]  = np.dot(inv_,cor_)

                # Reflexivo / Modo A
                if(Variables ['mode'][Variables['latent']==latent[i]]).any()=="A":
                    a = data_[Variables ['measurement'][Variables['latent']==latent[i]]]
                    cov_ = []
                    for j in range(len(a.columns)):
                        b = list(fscores.ix[:,latent[i]])
                        a_ = list(a.ix[:,j])
                        cov_.append( np.cov(a_,b)[0][1] )
                        
                    myindex = Variables ['measurement'][Variables['latent']==latent[i]]
                    myindex_ = latent[i]
                    outer_weights.ix[myindex.values, myindex_] = cov_

            fscores = pd.DataFrame.dot(data_,outer_weights)

            for i in range(len(latent)):
                outer_weights[latent[i]][Variables ['measurement'][Variables['latent']==latent[i]]] = outer_weights[latent[i]][Variables ['measurement'][Variables['latent']==latent[i]]]  / np.std(list(fscores.ix[:,latent[i]]))

            convergiu = 0
            diff_ = pd.DataFrame.sum(pd.DataFrame.sum(abs(outer_weights-last_outer_weights)))
            if (diff_ < (10**(-(self.stopCriterion)))):
                convergiu = 1
                break

        # END LOOP

        fscores = pd.DataFrame.dot(data_,outer_weights)

        # Outer Loadings

        outer_loadings = pd.DataFrame(0, index=manifests, columns=latent)

        for i in range(len(latent)):
            cor_ = []
            a = data_[Variables ['measurement'][Variables['latent']==latent[i]]]
            for j in range(len(a.columns)):
                b = list(fscores.ix[:,latent[i]])
                a_ = list(a.ix[:,j])
                cor, p = sp.stats.pearsonr(a_, b)
                cor_.append(cor)

            myindex = Variables ['measurement'][Variables['latent']==latent[i]]
            myindex_= latent[i]
            outer_loadings.ix[myindex.values, myindex_] = cor_

        path_matrix_low = copy.deepcopy(path_matrix)
        path_matrix_high = copy.deepcopy(path_matrix)
        path_matrix_range = copy.deepcopy(path_matrix)

        # Paths

        r2 = pd.DataFrame(0, index=np.arange(1), columns=latent)
        dependant=np.unique(LVariables.ix[:,'target'])

        for i in range(len(dependant)):
            independant=LVariables[LVariables.ix[:,"target"]==dependant[i]]["source"]
            dependant_ = fscores.ix[:,dependant[i]]
            independant_ = fscores.ix[:,independant]

            if (self.regression=='ols'):
                # Path Normal
                coef, resid = np.linalg.lstsq(independant_, dependant_)[:2]
                r2[dependant[i]] = 1 - resid / (dependant_.size * dependant_.var())

                path_matrix.ix[dependant[i],independant] = coef
                path_matrix_low.ix[dependant[i],independant] = 0
                path_matrix_high.ix[dependant[i],independant] = 0

            if (self.regression=='fuzzy'):
                ac, awL, awR = otimiza(dependant_, independant_, len(independant_.columns), self.h)
                path_matrix.ix[dependant[i],independant] = ac
                path_matrix_low.ix[dependant[i],independant] = awL
                path_matrix_high.ix[dependant[i],independant] = awR

        r2 = r2.T

        # Path Effects

        path_effects = [None] * len(latent)
        path_effects[0] = path_matrix

        for i in range(1,len(latent)):
            path_effects[i] = pd.DataFrame.dot(path_effects[i-1],path_matrix)
        for i in range(1,len(path_effects)):
            indirect_effects = indirect_effects + path_effects[i]

        total_effects = indirect_effects + path_matrix

        for i in range(len(path_matrix.columns)):
            for j in range(len(path_matrix.columns)):
                path_matrix_range.ix[i,j] = str(round(path_matrix_low.ix[i,j],3)) + ' ' + str(round(path_matrix_high.ix[i,j],3))

        self.path_matrix = path_matrix
        self.path_matrix_high = path_matrix_high
        self.path_matrix_low = path_matrix_low

        self.total_effects = total_effects.T
        self.indirect_effects = indirect_effects

        self.fscores = fscores
        self.outer_loadings = outer_loadings
        self.contador = contador
        self.convergiu = convergiu
        self.path_matrix_range = path_matrix_range
        self.outer_weights = outer_weights
        self.r2 = r2

    def impa(self):
        return self.runIMPA()

    def predict(self):
        return self.runPrediction()

    def srmr(self):
        return self.runSRMR()

    def scheme(self):
        return self.scheme

    def regression(self):
        return self.reg

    def implied(self):
        return self.runImplied()

    def empirical(self):
        return self.runEmpirical()

    def cr(self):
        return self.runCR()

    def htmt(self):
        return self.runHTMT()

    def r2(self):
        return self.r2

    def alpha(self):
        return self.runAlphas()

    def corLVs(self):
        return self.runCorLVs()

    def xloads(self):
        return self.runXloads()

    def rhoA(self):
        return self.runRhoA()

    def outer_loadings(self):
        return self.outer_loadings

    def outer_weights(self):
        return self.outer_weights

    def comunalidades(self):
        return self.runComunalidades()

    def AVE(self):
        return self.runAVE()

    def fscores(self):
        return self.fscores

    def path_matrix(self):
        return self.path_matrix

    def path_matrix_high(self):
        return self.path_matrix_high

    def path_matrix_low(self):
        return self.path_matrix_low

    def path_matrix_range(self):
        return self.path_matrix_range

    def contador(self):
        return self.contador

    def convergiu(self):
        return self.convergiu