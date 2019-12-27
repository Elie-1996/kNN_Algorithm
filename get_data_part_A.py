from mlxtend.data import loadlocal_mnist
import numpy


def read_files(images_path, labels_path, sub_indices_to_consider_path):
    data_set, labels = loadlocal_mnist(
        images_path=images_path,
        labels_path=labels_path
    )
    labels = labels[:, numpy.newaxis]
    with open(sub_indices_to_consider_path, "r") as fd:
        lines = fd.read().splitlines()
    lines = list(map(int, lines))
    data_set = data_set[lines]
    labels = labels[lines]
    return data_set, labels


def combine_and_shuffle_data(data_set1, labels1, data_set2, labels2):
    first = numpy.concatenate((data_set1, labels1), axis=1)
    second = numpy.concatenate((data_set2, labels2), axis=1)
    entire_labeled_data = numpy.concatenate((first, second), axis=0)

    # randomize locations ~by row only~
    numpy.random.shuffle(entire_labeled_data)
    return entire_labeled_data


def split_data_into_training_and_test_sets(entire_labeled_data, percentage: float):

    # split the data points and labels again
    number_of_columns = len(entire_labeled_data[0])
    last_column = number_of_columns - 1
    entire_data_set = entire_labeled_data[:, 0:last_column]
    entire_labeled_set = entire_labeled_data[:, last_column]

    # allocate the 80% random data points and 20% random test points
    number_of_rows = len(entire_labeled_data)
    cut_index = int(percentage*number_of_rows)

    training_data = entire_data_set[0:cut_index, :]
    training_labels = entire_labeled_set[0:cut_index]
    training_labels = training_labels[:, numpy.newaxis]  # add new axis so its a "2d array"
    test_data = entire_data_set[cut_index:number_of_rows, :]
    test_labels = entire_labeled_set[cut_index:number_of_rows]
    test_labels = test_labels[:, numpy.newaxis]  # add new axis so its a "2d array"

    return training_data, training_labels, test_data, test_labels


def get_training_and_test_set():
    data_set1, labels1 = read_files('training_set/train-images.idx3-ubyte', 'training_set/train-labels.idx1-ubyte', 'training_set/train_indices.txt')
    data_set2, labels2 = read_files('test_set/t10k-images.idx3-ubyte', 'test_set/t10k-labels.idx1-ubyte', 'test_set/test_indices.txt')
    entire_labeled_data = combine_and_shuffle_data(data_set1, labels1, data_set2, labels2)
    _training_set, _training_label_set, _test_data_set, _test_label_set = split_data_into_training_and_test_sets(
        entire_labeled_data,
        0.8
    )

    return _training_set, _training_label_set, _test_data_set, _test_label_set

