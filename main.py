from embedding import getGloVeModel
from compressor import compressEmbedding

import numpy as np

def getData():
    return data

def getEmbedding(data):
    embedding_index = getGloVeModel('data/glove.6B.50d.txt')
    # convert embedding matrix in dense format pertaining to dataset
    return compressEmbedding(embedding_index, data)

def splitData(data, split_value=0.5):
    """"
    splitData(data, split_value)

        split the data into a training set and a validation set
        split_value is fraction of samples used for validation set
    """

    indices = np.arange(data.X.shape[0])
    np.random.shuffle(indices)

    X = data.X[indices]
    y = data.y[indices]
    nb_validation_samples = int(split_value * X.shape[0])

    x_train = X[:-nb_validation_samples]
    y_train = y[:-nb_validation_samples]
    x_val = X[-nb_validation_samples:]
    y_val = y[-nb_validation_samples:]

    return x_train,y_train,x_val,y_val

data = getData()
word_index, data, embedding_matrix = getEmbedding(data)
x_train, y_train, x_val, y_val = splitData(data, split_value=0.3)