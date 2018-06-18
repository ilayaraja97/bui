import getopt
import re

import numpy as np
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json

from parseAmazon import parse_amazon, parse_amazon_large, parse_amazon_medium
from parseKaggle import parse_kaggle

"""
This file cleans, and embeds the data set. The embedding is done using 50d GloVe model. 
"""


def get_glove_model(file):
    """
    the function opens the pre-trained glove embedding downloaded from stanford - 50 dimensional glove embedding
    :param:Path to glove file.
    :return:Embedding index
    """
    embedding_index = {}

    with open(file, 'r', encoding='utf-8') as gloveFile:
        for line in gloveFile:
            line = line.split(' ')
            word = line[0]
            vector = np.asarray(line[1:], dtype='float32')
            embedding_index[word] = vector

    print('Found %s word vectors.' % len(embedding_index))
    return embedding_index


def get_embedding(data):
    """
    the pre-trained glove embedding is used to get the embedding index
    :param:(x_train, y_train)
    :return:(Vocabulary, (x_train, y_train), embedding_matrix)
    """
    embedding_index = get_glove_model('data/glove.6B.50d.txt')

    # get embedding_matrix by pre-processing the data.
    max_seq_len = 250

    # Create an instance of Tokenizer and convert text into sequences so as to pad the sequence
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(np.copy(data[0]))
    sequences = tokenizer.texts_to_sequences(data[0])

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    print('Example word index ', word_index.get('the'))
    print('Example sequence ', sequences[0])

    # get a 2D numpy array of input and output
    x = pad_sequences(sequences, maxlen=max_seq_len)
    y = data[1]
    embedding_dim = 50
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return word_index, (x, y), embedding_matrix


def split_data(data, split_value=0.5):
    """"
    split_data(data, split_value):
        split the data into a training set and a validation set
        split_value is fraction of samples used for validation set
    :param:data = x_train, y_train
    :param:split_value =  fraction of samples used for validation set
    :return:x_train, y_train, x_val, y_val
    """

    indices = np.arange(data[0].shape[0])
    np.random.shuffle(indices)

    X = data[0][indices]
    y = data[1][indices]
    nb_validation_samples = int(split_value * X.shape[0])

    x_train = X[:-nb_validation_samples]
    y_train = y[:-nb_validation_samples]
    x_val = X[-nb_validation_samples:]
    y_val = y[-nb_validation_samples:]

    return x_train, y_train, x_val, y_val


def get_encoded_matrix(vocab, data, max_seq_length):
    """
    Encode sentences in data to a sequence which is understandable by neural nets.
    :param:vocab is a dictionary which has the sequence for each word.
    If the word in data doesn't exist in vocab a 0 sequence is returned
    :return:A sequence matrix of the given data is returned
    """
    ids = np.zeros([len(data), max_seq_length], dtype=int)
    for i, sentence in enumerate(data):
        for j, word in enumerate(re.split("[ !\"#$%&*+,-./:;<=>?@^_`{|}~\t\n']", sentence)):
            if j == max_seq_length:
                break
            if word.lower() in vocab:
                # print(word)
                ids[i][j] = vocab.get(word.lower())
            else:
                ids[i][j] = 0
    return ids


def clean_data(size=0):
    """
    Cleans data, takes data from amazon dataset, encodes it and save in a numpy file.
    :return: void. But data is stored in 5 files in data folder.
    """
    x, y = parse_amazon()
    term = ''
    if size == 1:
        x, y = parse_amazon_medium()
        term = '_500k'
    elif size == 1:
        x, y = parse_amazon_large()
        term = '_1m'

    word_index, data, embedding_matrix = get_embedding((x, y))
    print("loaded")
    # validate with kaggle
    # x1, y1 = parse_kaggle()
    # x1 = get_encoded_matrix(dict(word_index), x1, 250)
    # word_index2, data2, embedding_matrix2 = get_embedding((x1, y1))
    # x_train, y_train, x_val, y_val = data[0], data[1], x1, y1

    # validate with amazon
    x_train, y_train, x_val, y_val = split_data(data, split_value=0.1)
    print("split")
    # print(x_train)
    # print(y_train)
    # print(x_val)
    # print(y_val)
    with open('data/word_index' + term + '.json', "w") as outf:
        json.dump(word_index, outf)
    np.save('data/embedding_matrix' + term, embedding_matrix)
    np.save('data/x_train' + term, x_train)
    np.save('data/y_train' + term, y_train)
    np.save('data/x_val' + term, x_val)
    np.save('data/y_val' + term, y_val)


def main(argv):
    global opts
    try:
        opts, args = getopt.getopt(argv, "hsml")
    except getopt.GetoptError:
        print('preProcessing.py -[sml]')
        sys.exit()
    for opt, arg in opts:
        if opt == '-h':
            print('preProcessing.py -[sml]')
            sys.exit()
        if opt == '-s':
            clean_data(size=0)
        elif opt == '-m':
            clean_data(size=1)
        elif opt == '-l':
            clean_data(size=2)


if __name__ == "__main__":
    main(sys.argv[1:])
