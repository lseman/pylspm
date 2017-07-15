# Adapted from https://github.com/log0ymxm/predictive_imputer

from numpy import inf
import pandas as pd
import numpy as np
from regForest import *
import pandas as pd

class Imputer(object):

    def __init__(self, max_iter=10):
        self.max_iter = max_iter

    def isNaN(self, num):
        return num != num

    def meanImput(self, X):
        X_ = pd.DataFrame(X)
        mean = pd.DataFrame.mean(X_)
        for j in range(len(X_.columns)):
            for i in range(len(X_)):
                if (self.isNaN(X_.ix[i, j])):
                    X_.ix[i, j] = mean[j]
        return np.array(X_)

    def fit(self, X, y=None, **kwargs):
        X = np.array(X)
        X_nan = np.isnan(X)
        most_by_nan = X_nan.sum(axis=0).argsort()[::-1]

        # Mean imputation as start
        imputed = self.meanImput(X.copy())
        new_imputed = imputed.copy()

        ntrees = 10
        params = {'max_depth': 10,
                  'min_sample_count': 2,
                  'test_count': 10}

        self.estimators_ = [RegressionForest(
            ntrees, params) for i in range(X.shape[1])]

        old_imputed = imputed.copy()
        old_gamma = np.inf

        old_estimators = []

        for iter in range(self.max_iter):
            print('Iteraration ' + str(iter + 1))
            for i in most_by_nan:
                print(i)
                X_s = np.delete(new_imputed, i, 1)
                y_nan = X_nan[:, i]

                X_train = X_s[~y_nan]
                y_train = new_imputed[~y_nan, i]
                X_unk = X_s[y_nan]

                estimator_ = self.estimators_[i]
                X_train = X_train
                y_train = y_train

                #print('x trainn')
                #print(X_train)
                #print('y train')
                #print(y_train)

                estimator_.fit(X_train, y_train)

                result_ = []
                if len(X_unk) > 0:
                    for unk in X_unk:
                        result_.append(estimator_.predict(unk))
                    new_imputed[y_nan, i] = result_

            gamma = np.sum(((new_imputed - old_imputed)**2) / (new_imputed**2))
            if ((gamma > old_gamma)):
                self.estimator_ = old_estimators.copy()
                break

            old_estimators = self.estimators_.copy()
            old_imputed = new_imputed.copy()
            old_gamma = gamma.copy()

#        print(new_imputed)

        return self

    def get(self, X):
        X = np.array(X)
        X_nan = np.isnan(X)
        imputed = self.meanImput(X.copy())

        if len(self.estimators_) > 1:
            for i, estimator_ in enumerate(self.estimators_):
                X_s = np.delete(imputed, i, 1)
                y_nan = X_nan[:, i]

                X_unk = X_s[y_nan]

                result_ = []
                if len(X_unk) > 0:
                    for unk in X_unk:
                        result_.append(estimator_.predict(unk))
                    X[y_nan, i] = result_

        return X
