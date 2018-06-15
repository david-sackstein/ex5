# -----------------------------------------------------------------------------
# Based on the code of  Martin Thoma
#           https://martin-thoma.com/classify-mnist-with-pybrain/
#
#
# Author: Erez Peterfreund , 
#         erezpeter@cs.huji.ac.il  , 2018
# 
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

from struct import unpack
import gzip
from numpy import zeros, uint8, float32
from matplotlib.pyplot import imshow
import numpy as np

# The function samples from the given data only the first samples
# with the given labels
# Input:
#           chosenLabels - A list with all the labels that you would like the samples to have
#           nOfEach - The amount of samples that you would like from each label
#           data - a n-by-d matrix with n samples of dimension d
#           labels- a d-by-1 matrix with d labels
#
# Output:
#           outData- a matrix with samples ordered as in chosenLabels. Each row conatins a sample.
#           outLabels- a matrix with the labels of each sample
def genMnistDigits(chosenLabels,nOfEach,data, labels):
    if (data is None) or (labels is None):
        data, labels = readMnist()
    
    outData= np.zeros([nOfEach*len(chosenLabels),784])
    outLabels= np.zeros([len(chosenLabels)*nOfEach,1])
    
    for i in range(len(chosenLabels)):
        label= chosenLabels[i]
        indexes= np.where(label==labels)[0]
        
        if len(indexes)<nOfEach:
            raise Exception("genMnistDigits: There are only "+str(len(indexes))+" samples of the label "+str(label)) 
        
        dataIndexes= indexes[range(nOfEach)]        
        outIndexes= range(nOfEach*i, nOfEach*(i+1))
        
        outData[outIndexes] = data[dataIndexes]
        outLabels[range(i*nOfEach,(i+1)*nOfEach)]= np.reshape(labels[dataIndexes],(-1,1))
        
    return outData,outLabels


def readMnist():
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """

    #imagefile= os.path.join(mnistPath,'train-images-idx3-ubyte.gz')
    #labelfile= os.path.join(mnistPath,'train-labels-idx1-ubyte.gz')
    imagefile= 'train-images-idx3-ubyte.gz'
    labelfile= 'train-labels-idx1-ubyte.gz'

    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array
    y = zeros(N, dtype=uint8)  # Initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i) # For python 3.
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (x.reshape(x.shape[0],x.shape[1]*x.shape[2]), y)



def showImage(im):
    if len(im.shape)==1:
        im=im.reshape([28,28])
    elif im.shape[0]==1 or im.shape[1]==1:
        im=im.reshape([28,28])        
    imshow(im)

