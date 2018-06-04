import numpy as np

def getGloVeModel(file):
    embedding_index = {}

    with open(file, 'r', encoding='utf-8') as gloveFile:
        for line in gloveFile:
            line = line.split(' ')
            word = line[0]
            vector = np.asarray(line[1:], dtype='float32')
            embedding_index[word] = vector

    print('Found %s word vectors.' % len(embedding_index))
    return embedding_index