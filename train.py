import getopt
import json

import sys
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, GRU, Conv1D, GlobalMaxPooling1D, Dropout
from keras.layers import LSTM
import numpy as np

from keras_contrib.layers.advanced_activations import *

from parseKaggle import parse_kaggle
from preProcessing import get_encoded_matrix

def save_model(model, index="", dataset=""):
    model_json = model.to_json()
    with open("data/model" + index + dataset + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("data/model" + index + dataset + ".h5")
    print("Saved model to disk")


def train(max_features=20000, maxlen=250, batch_size=128, dataset="", modelname="-cnn", epochs=8, activation=""):

    print('Loading data...')

    (x_train, y_train), (x_test, y_test) = (np.load("data/x_train" + dataset + ".npy"),
                                            np.load("data/y_train" + dataset + ".npy")), \
                                           (np.load("data/x_val" + dataset + ".npy"),
                                            np.load("data/y_val" + dataset + ".npy"))
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)
    # print(y_train)

    embedding_matrix = np.load("data/embedding_matrix" + dataset + ".npy")
    print("word vector dimension", len(embedding_matrix[0]))

    model = Sequential()
    if modelname[1:] == 'cnn':
        print('Build CNN model...')

        model.add(
            Embedding(len(embedding_matrix), len(embedding_matrix[0]), weights=[embedding_matrix], trainable=False,
                      input_length=maxlen))
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(256, 3, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print('Train CNN...')
    elif modelname[1:] == 'gru':
        print('Build model GRU...')

        model.add(
            Embedding(len(embedding_matrix), len(embedding_matrix[0]), weights=[embedding_matrix], trainable=False,
                      input_length=maxlen))
        model.add(SpatialDropout1D(0.2))
        model.add(GRU(256, dropout=0.2, recurrent_dropout=0.2))
        if activation == "act1":
            model.add(Act1())
        elif activation == "act2":
            model.add(Act2())
        elif activation == "act3":
            model.add(Act3())
        elif activation == "act4":
            model.add(Act4())
        elif activation == "act5":
            model.add(Act5())
        elif activation == "act8":
            model.add(Act8())
        elif activation == "act9":
            model.add(Act9())
        elif activation == "act10":
            model.add(Act10())
        elif activation == "act11":
            model.add(Act11())
        elif activation == "act12":
            model.add(Act12())

        model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print('Train GRU...')
    elif modelname[1:] == 'lstm':
        print('Build model LSTM...')

        model.add(
            Embedding(len(embedding_matrix), len(embedding_matrix[0]), weights=[embedding_matrix], trainable=False,
                      input_length=maxlen))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))

        if activation == "act1":
            model.add(Act1())
        elif activation == "act2":
            model.add(Act2())
        elif activation == "act3":
            model.add(Act3())
        elif activation == "act4":
            model.add(Act4())
        elif activation == "act5":
            model.add(Act5())
        elif activation == "act8":
            model.add(Act8())
        elif activation == "act9":
            model.add(Act9())
        elif activation == "act10":
            model.add(Act10())
        elif activation == "act11":
            model.add(Act11())
        elif activation == "act12":
            model.add(Act12())

        model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print('Train LSTM...')

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              verbose=2)
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    if activation is not "":
        print ("Testing")
        a,b = parse_kaggle()
        with open('data/word_index'+dataset+'.json') as f:
            word_index = json.load(f)
        x = get_encoded_matrix(dict(word_index), a, 250)
        y = model.predict(x,batch_size=1,verbose=1)

        c=0
        for i in range(len(y)):
            v = round(y[i][0])
            if v == b[i]:
                c=c+1
        print("Testing accuracy", c/len(y)*100, "%")
    save_model(model, index="-amazon" + modelname, dataset=dataset)


def main(argv):
    global opts
    try:
        opts, args = getopt.getopt(argv, "ho:smle:a:")
    except getopt.GetoptError:
        print('train.py -[sml] -e epochs -o model -a activation')
        sys.exit()

    dataset = ""
    modelname = "-cnn"
    epochs = 8
    activation = ""

    for opt, arg in opts:
        if opt == '-h':
            print('train.py -[sml] -e epochs -o model -a activation')
            sys.exit()
        if opt == '-s':
            dataset = ''
        elif opt == '-m':
            dataset = '_500k'
        elif opt == '-l':
            dataset = '_1m'
        if opt == "-e":
            epochs = int(arg)
        if opt == "-o":
            modelname = "-" + str(arg)
        if opt == "-a":
            activation = str(arg)
    print("train", modelname[1:], "for", epochs, "epochs on", dataset[1:], "dataset", "activation", activation)
    train(dataset=dataset, epochs=epochs, modelname=modelname, activation=activation)


if __name__ == "__main__":
    main(sys.argv[1:])
