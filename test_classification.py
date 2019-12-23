from get_data_part_A import get_training_and_test_set
import KnnClassifier as kNN

class tester:

    def __init__(self):
        # test 1:
        training_set = [[1], [2], [3], [4], [5], [6], [7]]
        training_label_set = [1, 1, 1, 9, 9, 9, 9]
        k = 3  # 3-Nearest-Neighbor
        classifier = kNN.KnnClassifier(k, training_set, training_label_set)
        print(classifier.classify([0]))
