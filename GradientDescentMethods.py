def GD(data, label, iters, eta):
    '''
        :param data: a n ˆ d matrix, where n is the amount of samples, and d is the dimension
        :param label: a n ˆ 1 matrix with the labels of each sample
        :param iters: an integer that will define the amount of iterations
        :param eta: a positive number that will define the learning rate
        :return: a d ˆ iters matrix, where in its i-th column it will contain the output of the
        sub-gradient descent algorithm over i iterations
    '''

    pass


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

    pass

def testError(w, testData, testLabels):
    '''
        :param w: a d ˆ l matrix, where l is the amount of the linear hypothesis that you would like to examine,
        each of which are defined using a d- dimensional vector (hi(xq) = <wi, x>, where wi is in Rd)
        :param testData: a n ˆ d matrix with the samples
        :param testLabels: a d ˆ 1 matrix with the labels of the samples
        :return: The function will output a lˆ1 matrix with the respective 0-1 loss for each hypothesis
        (Make sure to take the sign of the inner product as the predicted label).
    '''

    pass