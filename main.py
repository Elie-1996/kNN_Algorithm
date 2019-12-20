from mlxtend.data import loadlocal_mnist


def read_files(images_path, labels_path):
    data_set, labels = loadlocal_mnist(
        images_path=images_path,
        labels_path=labels_path
    )

    return data_set, labels


if __name__ == '__main__':
    data_set1, labels1 = read_files('test_set/t10k-images.idx3-ubyte', 'test_set/t10k-labels.idx1-ubyte')
    data_set2, labels2 = read_files('test_set/t10k-images.idx3-ubyte', 'test_set/t10k-labels.idx1-ubyte')


    # how to print the data in the files: (X is the data)
    # print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
    # print('\n1st row', X[0])
