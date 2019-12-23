import numpy
from math import sqrt
import heapq


class KnnClassifier:

    # k should be smaller than the number of training_set samples.
    def __init__(self, k: int, training_set, label_set):
        self.k: int = k
        self.training_set = training_set
        self.label_set = label_set

    @staticmethod
    def get_euclidean_distance(sample1, sample2):
        squared_differences: numpy.ndarray = ((sample1 - sample2)**2)  # x**2 is pow(x,2)
        return sqrt(squared_differences.sum())

    # find the k closest samples to test_sample, and return their k indices
    def get_k_nearest_training_samples(self, test_sample):

        heap = []
        for index, training_sample in enumerate(self.training_set):
            distance = self.get_euclidean_distance(training_sample, test_sample)
            heap.insert(index, (distance, index))
        heapq.heapify(heap)

        closest_k_indices = []
        for i in range(0, self.k):
            closest_k_indices.append(heapq.heappop(heap)[1])

        return closest_k_indices

    # classifies a real sample
    def classify(self, test_sample):
        sub_training_set_indices = self.get_k_nearest_training_samples(test_sample)
        indices = self.label_set[sub_training_set_indices]
        classes_counter = numpy.array([0] * 10)
        for index in indices:
            classes_counter[index] = classes_counter[index] + 1
        return numpy.argmax(classes_counter)  # the classification
