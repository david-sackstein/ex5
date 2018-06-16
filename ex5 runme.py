import os
import shutil

import numpy as np
from matplotlib.pyplot import plot, legend, ylabel, xlabel, title, savefig, clf, imsave
from GradientDescentMethods import GD, split_data_randomly, SGD, testError
from readMnist import readMnist, showImage

plot_dir = 'output/plots'
image_dir = 'output/images'

shutil.rmtree('output')
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

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


def plot_losses(name, hypotheses, train_data, train_labels, test_data, test_labels):
    losses_train_data = testError(hypotheses, train_data, train_labels)
    losses_test_data = testError(hypotheses, test_data, test_labels)

    hypothesis_count = hypotheses.shape[0]
    hypothesis_indexes = range(hypothesis_count)

    losses_train_data_line, = plot(hypothesis_indexes, losses_train_data, linestyle='--', label='Train data loss')
    losses_test_data_line, = plot(hypothesis_indexes, losses_test_data, linestyle='--', label='Test data loss')

    legend(handles=[losses_train_data_line, losses_test_data_line])

    ylabel('Loss')
    xlabel('[iteration]')

    title('Train Error and Test Error of {} as Functions at each Iteration'.format(name))
    file_name = '{}/Plot.{}.png'.format(plot_dir, name)
    savefig(file_name)
    clf()


def save_hypothesis_image(name, hypotheses):
    iters_to_print = {5, 15, 50, 100}
    for iter_ in iters_to_print:
        square = np.delete(hypotheses[iter_], -1).astype(np.uint8).reshape([28, 28])
        file_name = '{}/{}.iter.{}.png'.format(image_dir, name, iter_)
        imsave(file_name, square)


def plot_hypotheses(name, hypotheses, train_data, train_labels, test_data, test_labels):
    plot_losses(name, hypotheses, train_data, train_labels, test_data, test_labels)
    save_hypothesis_image(name, hypotheses)


if __name__ == '__main__':
    images_of_each_type = 500
    eta = 1
    iteration_count = 150
    image_types = [0, 1]
    batches = [5, 50, 150]

    raw_data, raw_labels = readMnist(image_types, images_of_each_type)

    data, labels = prepare_data(raw_data, raw_labels)

    train_data, train_labels, test_data, test_labels = split_data(data, labels)

    gd_hypotheses = GD(train_data, train_labels, iteration_count, eta)
    plot_hypotheses('GD', gd_hypotheses, train_data, train_labels, test_data, test_labels)

    for batch in batches:
        sgd_hypotheses = SGD(train_data, train_labels, iteration_count, eta, batch)
        name = 'SGD.batch.{}'.format(batch)
        plot_hypotheses(name, sgd_hypotheses, train_data, train_labels, test_data, test_labels)
