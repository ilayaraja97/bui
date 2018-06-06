from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np


def compressEmbedding(embedding_index, data):
    print("compress Embedding")
    MAX_SEQUENCE_LENGTH = 250
    # MAX_NB_WORDS = 72000

    tokenizer = Tokenizer()
    # print(np.copy(data[0]).shape)
    tokenizer.fit_on_texts(np.copy(data[0]))
    sequences = tokenizer.texts_to_sequences(data[0])

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    print('Example word index ', word_index.get('the'))
    print('Example sequence ', sequences[0])

    # get a 2D numpy array of input and output
    x = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y = data[1]
    print(y.shape)
    EMBEDDING_DIM = 50
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # print(len(embedding_matrix), " ", embedding_matrix[word_index.get('and')])

    return word_index, (x, y), embedding_matrix
