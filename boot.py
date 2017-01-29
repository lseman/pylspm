# PyLS-PM bootstraping Library
# Author: Laio Oriel Seman
# Creation: November 2016

from multiprocessing import Pool, freeze_support
import pandas as pd
import numpy as np
from pylspm import PyLSpm


class PyLSboot(object):

    def do_work(self, item):
        amostra = self.data.sample(
            len(self.data), replace=True, random_state=(np.random.RandomState()))
        amostra.index = range(len(self.data))
        bootstraping = PyLSpm(amostra, self.LVcsv, self.Mcsv, self.scheme,
                              self.reg, self.h, self.maximo, self.stopCriterion)
        if (bootstraping.convergiu == 1):
            return [bootstraping.path_matrix.values]

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

                sumOuterResid = pd.DataFrame.sum(
                    pd.DataFrame.sum(results[i].residuals()[1]**2))
                sumInnerResid = pd.DataFrame.sum(
                    pd.DataFrame.sum(results[i].residuals()[2]**2))
                f1.append(sumOuterResid + sumInnerResid)
            except:
                f1.append(10000)
        print((1 / np.sum(f1)))
        return (1 / np.sum(f1))

    def __init__(self, br, cores, dados, LVcsv, Mcsv, scheme='path', reg='ols', h=0, maximo=300, stopCrit=7, nclusters=2, population='none'):

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

    def gac(self):
        p = Pool(self.cores)
        result = p.map(self.do_work_ga, range(self.br))
        p.close()
        p.join()
        return result
