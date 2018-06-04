from embedding import getGloVeModel
from compressor import compressEmbedding


def getData():
    return data

def getEmbedding(data):
    embedding_index = getGloVeModel('data/glove.6B.50d.txt')
    # convert embedding matrix in dense format pertaining to dataset
    return compressEmbedding(embedding_index, data)

