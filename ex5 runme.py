from readMnist import genMnistDigits, readMnist

if __name__ == '__main__':

    data, labels = readMnist()
    currData, currLabels = genMnistDigits([0, 1], 500, data, labels)

    pass
