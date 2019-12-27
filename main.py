from get_data_part_A import get_training_and_test_set
import KnnClassifier as kNN
from math import sqrt


def get_classification_error(classified_set, actual_classification):
    total = 0
    incorrect = 0
    for index, result in enumerate(classified_set):
        total = total + 1
        if result != actual_classification[index]:
            incorrect = incorrect + 1

    return incorrect / total


if __name__ == '__main__':
    training_set, training_label_set, test_data_set, test_label_set = get_training_and_test_set()
    k = 4  # int(sqrt(training_set.shape[0]))  # 3-Nearest-Neighbor
    classifier = kNN.KnnClassifier(k, training_set, training_label_set)

    classified_set = []
    counter = 0
    for sample_point in test_data_set:
        classification = classifier.classify(sample_point)
        classified_set.append(classification)
        error_rate = get_classification_error(classified_set, test_label_set) * 100.0
        counter = counter + 1
        print(counter.__str__() + ': (k = ' + k.__str__() + ') ' + error_rate.__str__())

    print("~~~~~~~~~~~")
    error_rate = get_classification_error(classified_set, test_label_set) * 100.0
    print(error_rate)


