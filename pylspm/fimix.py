# HAHN, C. et al. Capturing Customer Heterogeneity Using a Finite Mixture
# PLS Approach. Schmalenbach Business Review, v. 54, n. July, p. 243–269,
# 2002.

# SARSTEDT, M. et al. Uncovering and Treating Unobserved Heterogeneity
# with FIMIX-PLS: Which Model Selection Criterion Provides an Appropriate
# Number of Segments? Schmalenbach Business Review, v. 63, n. 1, p. 34–62,
# 2011.

import pandas
import numpy as np
from numpy import inf
import pandas as pd
import scipy.stats
from scipy.stats import norm

from .pylspm import PyLSpm
from .boot import PyLSboot
from .bootstraping import bootstrap


class fimixPLS(object):

    def weighted_linear_regression(self, x, y, weights):
        return np.linalg.inv((weights[:, np.newaxis] * x).T.dot(x)).dot((weights[:, np.newaxis] * x).T.dot(y))

    def weighted_regression_variance(self, x, y, weights, coefficients):
        result = 0.0
        x = x.values
        y = y.values
        for i in range(len(y)):
            result += weights[i] * (y[i] - x[i].T.dot(coefficients)) ** 2
        return result / weights.sum()

    # M-step (maximum likelihood)
    def maximization(self, fscores, assignments, assignment_weights):

        coefficients = []
        variances = []

        dependent = np.unique(self.LVariables.ix[:, 'target'])

        data = []
        for i in range(self.num_components):
            data_ = (fscores.loc[fscores['Split'] == i]).drop('Split', axis=1)
            data_.index = range(len(data_))
            print(len(data_))
            data.append(data_)

        varDiag = []

        for j in range(self.num_components):
            coefficients_ = self.path_matrix.copy()
            variances_ = self.path_matrix.copy()
            varDiag_ = []

            data_ = data[j]

            points = np.where(assignments == j)[0]
            subset_weights = assignment_weights[points][:, j]
#            print(subset_weights)

            for i in range(len(dependent)):
                independent = self.LVariables[self.LVariables.ix[
                    :, "target"] == dependent[i]]["source"]
                dependent_ = data_.ix[:, dependent[i]]
                independent_ = data_.ix[:, independent]

                coef, resid = np.linalg.lstsq(independent_, dependent_)[:2]
                coef_ = self.weighted_linear_regression(
                    independent_, dependent_, subset_weights)

                coefficients_.ix[dependent[i], independent] = coef_

                var_ = self.weighted_regression_variance(
                    independent_, dependent_, subset_weights, coef_)

                variances_.ix[dependent[i], independent] = var_
                varDiag_.append(var_)

            varDiag.append(np.diag(varDiag_))

            coefficients.append(coefficients_)
            variances.append(variances_)

        return coefficients, varDiag

    # E-step
    def expectation(self, dataSplit, coefficients, variances):

        assignment_weights = np.ones(
            (len(dataSplit), self.num_components), dtype=float)

        self.Q = len(self.endoVar)

        for k in range(self.num_components):

            coef_ = coefficients[k]

            Beta = coef_.ix[self.endoVar][self.endoVar]
            Gamma = coef_.ix[self.endoVar][self.exoVar]

            a_ = (np.dot(Beta, self.fscores[
                  self.endoVar].T) + np.dot(Gamma, self.fscores[self.exoVar].T))

            invert_ = np.linalg.inv(np.array(variances[k]))

            exponential = np.exp(-0.5 * np.dot(np.dot(a_.T, invert_), a_))

            den = (((2 * np.pi)**(self.Q / 2)) *
                   np.sqrt(np.linalg.det(variances[k])))
            probabilities = exponential / den
            probabilities = probabilities[0]
            assignment_weights[:, k] = probabilities

        assignment_weights /= assignment_weights.sum(axis=1)[:, np.newaxis]
 #       print(assignment_weights)
        return assignment_weights

    def calculate_assignments(self, assignment_weights):
        clusters = np.argmax(assignment_weights, axis=1)
        return clusters

    def data_log_likelihood(self, dataSplit, coefficients, variances):

        log_likelihood = 0.0

        for k in range(self.num_components):

            coef_ = coefficients[k]

            Beta = coef_.ix[self.endoVar][self.endoVar]
            Gamma = coef_.ix[self.endoVar][self.exoVar]

            a_ = (np.dot(Beta, self.fscores[
                  self.endoVar].T) + np.dot(Gamma, self.fscores[self.exoVar].T))

            invert_ = np.linalg.inv(np.array(variances[k]))

            exponential = np.exp(-0.5 * np.dot(np.dot(a_.T, invert_), a_))

            den = (((2 * np.pi)**(self.Q / 2)) *
                   np.sqrt(np.linalg.det(variances[k])))
            probabilities = exponential[0] / den

            log_likelihood += np.log(probabilities).sum()

        print(log_likelihood)
        return log_likelihood

    def __init__(self, num_components, data_, lvmodel, mvmodel, scheme, regression, h='0', maxit='100'):

        self.num_components = num_components
        threshold = 1e-7
        scores = PyLSpm(data_, lvmodel, mvmodel, scheme,
                        regression, 0, 100, HOC='false', disattenuate='false')
        self.LVariables = scores.LVariables
        fscores = scores.fscores
        self.fscores = fscores
        self.path_matrix = pd.DataFrame(
            0, index=scores.latent, columns=scores.latent)

        self.endoVar, self.exoVar = scores.endoexo()

        self.Q = len(self.endoVar)

        prev_log_likelihood = 1
        cur_log_likelihood = 0

        # Random start
        assignment_weights = np.random.uniform(
            size=(len(fscores), self.num_components))

        assignment_weights /= assignment_weights.sum(axis=1)[:, np.newaxis]
        assignments = self.calculate_assignments(assignment_weights)

        clusters = pd.DataFrame(assignments)
        clusters.columns = ['Split']
        dataSplit = pd.concat([fscores, clusters], axis=1)

        coefficients, variances = self.maximization(
            dataSplit, assignments, assignment_weights)

        iteration = 0
        while np.abs(prev_log_likelihood - cur_log_likelihood) > threshold:

            assignment_weights = self.expectation(
                fscores, coefficients, variances)

            assignments = self.calculate_assignments(assignment_weights)
            clusters = pd.DataFrame(assignments)
            clusters.columns = ['Split']
            dataSplit = pd.concat([fscores, clusters], axis=1)

            coefficients, variances = self.maximization(
                dataSplit, assignments, assignment_weights)

            prev_log_likelihood = np.copy(cur_log_likelihood)

            cur_log_likelihood = self.data_log_likelihood(
                fscores, coefficients, variances)

            iteration += 1

        K = self.num_components
        R = len(self.LVariables)
        NK = (K - 1) + K * R + K * self.Q
        AIC = -2 * (cur_log_likelihood) + 2 * NK
        print('AIC')
        print(AIC)

        print('BIC')
        BIC = -2 * (cur_log_likelihood) + np.log(len(dataSplit)) * NK
        print(BIC)
