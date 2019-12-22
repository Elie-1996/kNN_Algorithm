from numpy import ndarray, sum
from math import sqrt


class KnnClassifier:

    def __init__(self, k, training_set, label_set):
        self.k = k
        self.training_set = training_set
        self.label_set = label_set
        self.classify()

    @staticmethod
    def get_euclidean_distance(sample1, sample2):
        squared_differences: ndarray = ((sample1 - sample2)**2)
        return sqrt(squared_differences.sum())

    # code the classification
    def classify(self):
        pass
