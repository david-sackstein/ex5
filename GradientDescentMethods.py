import numpy as np


def hinge_sample_loss(label, prediction):
    return max(0, 1 - prediction * label)


def zero_one_sample_loss(label, prediction):
    return 0 if prediction * label > 0 else 1


def sample_sub_grad(datum, label, prediction):
    dimension = len(datum)
    sub_grad = np.zeros(dimension)
    for i in range(dimension):
        sub_grad[i] = 0 if label * prediction >= 1 else - label * datum[i]
    return sub_grad


def get_average_sub_grad(w, data, label):

    sample_count = data.shape[0]
    dimension = data.shape[1]

    average_sub_grad = np.zeros(dimension)

    for i in range(sample_count):
        prediction = np.inner(data[i], w)
        sub_grad = sample_sub_grad(data[i], label[i], prediction)
        average_sub_grad += sub_grad

    return average_sub_grad / sample_count


def split_data_randomly(total_size, selection_size):
    # create the other_indexes has having all the indexes
    # and the selected_indexes as being empty.
    # We will move selection_size indexes from other_indexes to selected_indexes

    other_indexes = [i for i in range(total_size)]
    selected_indexes = []

    for i in range(selection_size):
        # we need to remove one of the indexes in other_indexes.
        # but note that other_indexes may contain gaps because we have removed some of its original values.
        # So for instance, if other_indexes now contains [0, 55, 2345, 20000]
        # and we need to remove one more we select a position between 0 and 3 - let's say 1
        # and then we remove 55 which is in place 1 (and move it to selected_indexes)

        select_position_of_index_to_move = np.random.randint(0, len(other_indexes))

        index_to_move = other_indexes[select_position_of_index_to_move]
        other_indexes.remove(index_to_move)
        selected_indexes.append(index_to_move)

    return selected_indexes, other_indexes


def select_batch(data, label, batch):
    sample_count = data.shape[0]
    if batch == sample_count:
        return data, label

    selected_indexes, _ = split_data_randomly(data.shape[0], batch)
    batch_data = [data[i] for i in selected_indexes]
    batch_label = [label[i] for i in selected_indexes]
    return np.asarray(batch_data), np.asarray(batch_label)

def GD(data, label, iters, eta):
    '''
        :param data: n x d matrix, where n is the amount of samples, and d is the dimension
        :param label: n x 1 matrix with the labels of each sample
        :param iters: integer that will define the amount of iterations
        :param eta: positive number that will define the learning rate
        :return: d x iters matrix, where in its i-th column it will contain the output of the
        sub-gradient descent algorithm over i iterations
    '''

    return SGD(data, label, iters, eta, data.shape[0])

def SGD(data, label, iters, eta, batch):

    '''
        :param data: a n ˆ d matrix, where n is the amount of samples, and d is the dimension
        :param label: a n ˆ 1 matrix with the labels of each sample
        :param iters: an integer that will define the amount of iterations
        :param eta: a positive number that will define the learning rate
        :param batch: The amount of samples that the algorithm would draw and use at each iteration
        :return: a d ˆ iters matrix, where in its i-th column it will contain the output of the sub-gradient
        descent algorithm over i iterations
    '''

    dimension = data.shape[1]
    w = np.zeros(dimension)
    all_weights = np.ndarray([iters, dimension])

    for iter_ in range(iters):
        batch_data, batch_label = select_batch(data, label, batch)
        average_sub_grad = get_average_sub_grad(w, batch_data, batch_label)
        w = w - eta * average_sub_grad
        all_weights[iter_] = w

    return all_weights


def testError(w, testData, testLabels):
    '''
        :param w: d x l matrix, where l is the amount of the linear hypothesis that you would like to examine,
        each of which are defined using a d- dimensional vector (hi(xq) = <wi, x>, where wi is in Rd)
        :param testData: n x d matrix with the samples
        :param testLabels: d x 1 matrix with the labels of the samples
        :return: The function will output a lx1 matrix with the respective 0-1 loss for each hypothesis
        (Make sure to take the sign of the inner product as the predicted label).
    '''

    hypothesis_count = w.shape[0]
    sample_count = testData.shape[0]

    errors = np.zeros(hypothesis_count)

    for i in range(hypothesis_count):
        hypothesis_weights = w[i]
        hypothesis_loss = 0
        for j in range(sample_count):
            sample = testData[j]
            label = testLabels[j]
            prediction = np.inner(hypothesis_weights, sample)
            hypothesis_loss += zero_one_sample_loss(label, prediction)
        errors[i] = hypothesis_loss

    return errors