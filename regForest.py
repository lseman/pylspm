# Adapted from https://github.com/kevin-keraudren/randomforest-python

import numpy as np
from multiprocessing import Pool, freeze_support
from numpy.random import uniform, random_integers

class RegressionTree(object):

    def __init__(self,
                 params={'max_depth': 10,
                         'min_sample_count': 2,
                         'test_count': 10}):
        self.params = params
        self.leaf = None
        self.left = None
        self.right = None
        self.test = None

    def MSE(self, responses):
        mean = np.mean(responses, axis=0)
        return np.mean((responses - mean) ** 2)

    def split_points(self, points, responses, test):
        left = []
        right = []
        for p, r in zip(points, responses):
            if (p[int(test[0])] > test[1]):
                right.append(r)
            else:
                left.append(r)
        return left, right

    def make_leaf(self, responses):
        self.leaf = np.mean(responses, axis=0)

    def generate_all(self, points, count):
        x_min = points.min(0)[0]
        x_max = points.max(0)[0]
        tests = []
        tests.extend(zip(np.zeros(count, dtype=int),
                         uniform(x_min, x_max, count)))
        return np.array(tests)

    def fit(self, points, responses, depth=0):

        error = self.MSE(responses)

        if (depth == self.params['max_depth']
                or len(points) <= self.params['min_sample_count']
                or error == 0):
            self.make_leaf(responses)
            return

        all_tests = self.generate_all(points, self.params['test_count'])

        best_error = np.inf
        best_i = None
        for i, test in enumerate(all_tests):
            left, right = self.split_points(points, responses, test)
            error = (len(left) / len(points) * self.MSE(left)
                     + len(right) / len(points) * self.MSE(right))
            if error < best_error:
                best_error = error
                best_i = i

        if best_i is None:
            self.make_leaf(responses)
            return

        self.test = all_tests[best_i]
        left_points = []
        left_responses = []
        right_points = []
        right_responses = []
        for p, r in zip(points, responses):
            if (p[int(self.test[0])] > self.test[1]):
                right_points.append(p)
                right_responses.append(r)
            else:
                left_points.append(p)
                left_responses.append(r)
        self.left = RegressionTree(self.params)
        self.right = RegressionTree(self.params)

        self.left.fit(np.array(left_points), left_responses, depth + 1)
        self.right.fit(np.array(right_points), right_responses, depth + 1)

    def predict(self, point):
        point = point.flatten()
        if self.leaf is not None:
            return self.leaf
        else:
            if (point[int(self.test[0])] > self.test[1]).any():
                return self.right.predict(point)
            else:
                return self.left.predict(point)


class RegressionForest(object):

    def __init__(self,
                 ntrees=10,
                 tree_params={'max_depth': 10,
                              'min_sample_count': 2,
                              'test_count': 10}):
        self.ntrees = ntrees
        self.tree_params = tree_params
        self.trees = []

    def fit(self, points, responses):
        for i in range(self.ntrees):
            self.trees.append(RegressionTree(self.tree_params))
            self.trees[i].fit(points, responses)

    def predict(self, point):
        response = []
        for i in range(self.ntrees):
            response.append(self.trees[i].predict(point))
        return np.mean(response, axis=0)
