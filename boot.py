# PyLS-PM bootstraping Library
# Author: Laio Oriel Seman
# Creation: November 2016

from multiprocessing import Pool, freeze_support
import pandas as pd
import numpy as np
from pylspm import PyLSpm
import random
from scipy.stats.stats import pearsonr


class PyLSboot(object):

    def do_work(self, item):
        amostra = self.data.sample(
            len(self.data), replace=True, random_state=(np.random.RandomState()))
        amostra.index = range(len(self.data))
        try:
            bootstraping = PyLSpm(amostra, self.LVcsv, self.Mcsv, self.scheme,
                                  self.reg, self.h, self.maximo, self.stopCriterion)
            if (bootstraping.convergiu == 1):
                return [bootstraping.path_matrix.values]
        except:
            return None

    def do_work_jk(self, item):
        amostra = self.data.ix[self.indices[item]].reset_index(drop=True)
        bootstraping = PyLSpm(amostra, self.LVcsv, self.Mcsv, self.scheme,
                              self.reg, self.h, self.maximo, self.stopCriterion)
        if (bootstraping.convergiu == 1):
            return [bootstraping.path_matrix.values]

    def do_work_ga(self, item):
        output = pd.DataFrame(self.population[item].genes)
        output.columns = ['Split']
        dataSplit = pd.concat([self.data, output], axis=1)
        f1 = []
        results = []
        for i in range(self.nclusters):
            dataSplited = (dataSplit.loc[dataSplit['Split']
                                         == i]).drop('Split', axis=1)
            dataSplited.index = range(len(dataSplited))

            try:
                results.append(PyLSpm(dataSplited, self.LVcsv, self.Mcsv, self.scheme,
                                      self.reg, 0, 50, HOC='true'))

                resid = results[i].residuals()[3]
                f1.append(resid)
            except:
                f1.append(10000)
        print((1 / np.sum(f1)))
        return (1 / np.sum(f1))

    def do_work_pso(self, item):
        output = pd.DataFrame(self.population[item].position)
        output.columns = ['Split']
        dataSplit = pd.concat([self.data, output], axis=1)
        f1 = []
        results = []
        for i in range(self.nclusters):
            dataSplited = (dataSplit.loc[dataSplit['Split']
                                         == i]).drop('Split', axis=1)
            dataSplited.index = range(len(dataSplited))

            try:
                results.append(PyLSpm(dataSplited, self.LVcsv, self.Mcsv, self.scheme,
                                      self.reg, 0, 50, HOC='true'))

                resid = results[i].residuals()[3]
                f1.append(resid)
            except:
                f1.append(10000)
        print((1 / np.sum(f1)))
        return (1 / np.sum(f1))

    def do_work_tabu(self, item):
        output = pd.DataFrame(self.population[item])
        output.columns = ['Split']
        dataSplit = pd.concat([self.data, output], axis=1)
        f1 = []
        results = []
        for i in range(self.nclusters):
            dataSplited = (dataSplit.loc[dataSplit['Split']
                                         == i]).drop('Split', axis=1)
            dataSplited.index = range(len(dataSplited))

            try:
                results.append(PyLSpm(dataSplited, self.LVcsv, self.Mcsv, self.scheme,
                                      self.reg, 0, 50, HOC='true'))

                resid = results[i].residuals()[3]
                f1.append(resid)
            except:
                f1.append(10000)

        cost = (np.sum(f1))
        print(1 / cost)
        return [self.population[item], cost]

    def do_work_permuta(self, item):

        node = np.zeros(self.lenT)
        while np.count_nonzero(node) != self.leng2:
            node[random.randint(0, self.lenT - 1)] = 1
        output = pd.DataFrame(node)
        output.columns = ['Split']
        dataSplit = pd.concat([self.dataPermuta, output], axis=1)

        results = []
        f1 = []
        f2 = []
        f3 = []

        try:
            for i in range(2):
                dataSplited = (dataSplit.loc[dataSplit['Split']
                                             == i]).drop('Split', axis=1)
                dataSplited.index = range(len(dataSplited))

                results.append(PyLSpm(dataSplited, self.LVcsv, self.Mcsv,
                                      self.scheme, self.reg, 0, 50, HOC='false'))
                outer_weights = results[i].outer_weights
                f1.append(outer_weights)

            singleResult = PyLSpm(self.dataPermuta, self.LVcsv,
                                  self.Mcsv, self.scheme, self.reg, 0, 50, HOC='false')
            fscores = singleResult.fscores

            for i in range(2):
                f2_ = fscores.loc[(dataSplit['Split'] == i)]
                f2.append(np.mean(f2_))
                f3.append(np.var(f2_))

            score1 = pd.DataFrame.dot(results[0].normaliza(dataSplited), f1[0])
            score2 = pd.DataFrame.dot(results[0].normaliza(dataSplited), f1[1])

            c = []
            for i in range(len(score1.columns)):
                c_ = np.corrcoef(score1.ix[:, i], score2.ix[:, i])
                c.append(c_[0][1])

            mean_diff = f2[0] - f2[1]

            log_diff = np.log(f3[0]) - np.log(f3[1])

    #        print(log_diff.values)
            return c, mean_diff, log_diff

        except:
            return None

    def __init__(self, br, cores, dados, LVcsv, Mcsv, scheme='path', reg='ols', h=0, maximo=300, stopCrit=7, nclusters=2, population=None, g1=None, g2=None, segmento=None):

        self.data = dados
        self.LVcsv = LVcsv
        self.Mcsv = Mcsv
        self.maximo = maximo
        self.stopCriterion = stopCrit
        self.h = h
        self.br = br
        self.cores = cores
        self.scheme = scheme
        self.reg = reg
        self.nclusters = nclusters
        self.population = population
        self.g1 = g1
        self.g2 = g2
        self.segmento = segmento

    def boot(self):
        p = Pool(self.cores)
        result = p.map(self.do_work, range(self.br))
        p.close()
        p.join()
        return result

    def jk(self):
        p = Pool(self.cores)

        base = np.arange(0, len(self.data))
        self.indices = list(np.delete(base, i) for i in base)

        result = p.map(self.do_work_jk, range(self.br))
        p.close()
        p.join()
        return result

    def permuta(self):

        self.dataPermuta = (self.data.loc[(self.data[self.segmento] == self.g1) | (
            self.data[self.segmento] == self.g2)]).drop(self.segmento, axis=1)
        self.dataPermuta.index = range(len(self.dataPermuta))

        self.leng1 = len(self.data.loc[(self.data[self.segmento] == self.g1)])
        self.leng2 = len(self.data.loc[(self.data[self.segmento] == self.g2)])
        self.lenT = self.leng1 + self.leng2

        p = Pool(self.cores)
        result = p.map(self.do_work_permuta, range(self.br))
        p.close()
        p.join()
        return result

    def gac(self):
        p = Pool(self.cores)
        result = p.map(self.do_work_ga, range(self.br))
        p.close()
        p.join()
        return result

    def pso(self):
        p = Pool(self.cores)
        result = p.map(self.do_work_pso, range(self.br))
        p.close()
        p.join()
        return result

    def tabu(self):
        p = Pool(self.cores)
        result = p.map(self.do_work_tabu, range(self.br))
        p.close()
        p.join()
        return result
