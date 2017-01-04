# PyLS-PM bootstraping Library
# Author: Laio Oriel Seman
# Creation: November 2016

from multiprocessing import Pool, freeze_support
import pandas
import numpy as np
from pylspm import PyLSpm

class PyLSboot(object):

    def do_work(self, item):
        amostra = self.data.sample(len(self.data), replace=True, random_state=item)
        amostra.index = range(len(self.data))
        bootstraping = PyLSpm(amostra, self.LVcsv, self.Mcsv, self.h, self.scheme, self.reg, self.maximo, self.stopCriterion)
        if (bootstraping.convergiu == 1):
#            return [bootstraping.outer_loadings.values, bootstraping.path_matrix.values, bootstraping.path_matrix_low.values, bootstraping.path_matrix_high.values]
            return [bootstraping.path_matrix.values]

    def do_work_jk(self, item):
        amostra = self.data.ix[self.indices[item]].reset_index(drop=True)
        bootstraping = PyLSpm(amostra, self.LVcsv, self.Mcsv, self.h, self.maximo, self.stopCriterion)
        if (bootstraping.convergiu == 1):
#            return [bootstraping.outer_loadings.values, bootstraping.path_matrix.values, bootstraping.path_matrix_low.values, bootstraping.path_matrix_high.values]
            return [bootstraping.path_matrix.values]

    def __init__(self, br, cores, dados, LVcsv, Mcsv, reg='ols', h=0, maximo=300, stopCrit=7):

        self.data = pandas.read_csv('dados2.csv')
        self.LVcsv = LVcsv
        self.Mcsv = Mcsv
        self.maximo = maximo
        self.stopCriterion = stopCrit
        self.h = h
        self.br = br
        self.cores = cores
        self.scheme = scheme
        self.reg = reg

    def boot(self):
        p = Pool(self.cores)
        result = p.map(self.do_work, range(self.br))
        p.close()
        p.join()
        return result

    def jk(self):
        p = Pool(self.cores)

        base = np.arange(0,len(self.data))
        self.indices = list(np.delete(base,i) for i in base)

        result = p.map(self.do_work_jk, range(self.br))
        p.close()
        p.join()
        return result