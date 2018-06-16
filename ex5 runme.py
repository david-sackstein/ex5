import numpy as np
from GradientDescentMethods import GD, split_data_randomly
from readMnist import readMnist, showImage

def show_images(raw_data, raw_labels):
    i = 0
    for image in raw_data:
        showImage(image)
        print(raw_labels[i])
        i += 1

def prepare_data(raw_data, raw_labels):
    raw_shape = raw_data.shape
    row_count = raw_shape[0]
    dimension = raw_shape[1]
    data = np.ndarray([row_count, dimension + 1])
    labels = np.ndarray(row_count)
    for i in range(row_count):
        data[i] = np.append(raw_data[i], 1)
        labels[i] = 1 if raw_labels[i] == 1 else -1
    return data, labels


def split_data(data, labels):

    total_size = data.shape[0]
    train_size = int(0.9 * total_size)

    train_indexes, test_indexes = split_data_randomly(total_size, train_size)

    train_data = [data[i] for i in train_indexes]
    train_labels = [labels[i] for i in train_indexes]
    test_data = [data[i] for i in test_indexes]
    test_labels = [labels[i] for i in test_indexes]

    return np.asarray(train_data), np.asarray(train_labels), \
           np.asarray(test_data), np.asarray(test_labels)

if __name__ == '__main__':

    images_of_each_type = 50
    image_types = [0, 1]

    raw_data, raw_labels = readMnist(image_types, images_of_each_type)

    # show_images(raw_data, raw_labels)
    data, labels = prepare_data(raw_data, raw_labels)

    train_data, train_labels, test_data, test_labels = split_data(data, labels)

    result = GD(train_data, train_labels, 10, 1)


