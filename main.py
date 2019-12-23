from get_data_part_A import get_training_and_test_set
import KnnClassifier as kNN
import time

if __name__ == '__main__':
    training_set, training_label_set, test_data_set, test_label_set = get_training_and_test_set()
    k = 3  # 3-Nearest-Neighbor
    classifier = kNN.KnnClassifier(k, training_set, training_label_set)

    samples_amount = test_data_set.shape[0]
    start_time = time.time()
    print(classifier.classify(test_data_set[0]))
    elapsed_time = time.time() - start_time
    print("Estimated time for ALL samples:")
    print(elapsed_time*samples_amount)

