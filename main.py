from get_data_part_A import get_training_and_test_set
import KnnClassifier as kNN


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

    lowest_error_and_k = (100.0, 0)
    for k in range(1, 15):
        classifier = kNN.KnnClassifier(k, training_set, training_label_set)

        classified_set = []
        for sample_point in training_label_set:
            classification = classifier.classify(sample_point)
            classified_set.append(classification)
        error_rate = get_classification_error(classified_set, training_label_set) * 100.0
        print('[validation] k = ' + k.__str__() + ': error = ' + error_rate.__str__())
        if error_rate < lowest_error_and_k[0]:
            lowest_error_and_k = (error_rate, k)

    print("Lowest:")
    print(lowest_error_and_k)

    # validation on test_set
    lowest_k = lowest_error_and_k[1]
    classified_set = []
    classifier = kNN.KnnClassifier(lowest_k, training_set, training_label_set)
    for sample_point in test_data_set:
        classification = classifier.classify(sample_point)
        classified_set.append(classification)
    error_rate = get_classification_error(classified_set, test_label_set) * 100.0
    print("[test] on k = " + lowest_k.__str__() + ", error = " + error_rate.__str__())


