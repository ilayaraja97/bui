import getopt
import json

import numpy as np
import sys
from keras.models import model_from_json

from parseKaggle import parse_kaggle
from preProcessing import get_encoded_matrix


def test(a, modelname="gru", dataset=""):
    """

    :param modelname:
    :return:
    """
    json_file = open('data/model-amazon' + modelname + dataset + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights('data/model-amazon' + modelname + dataset + '.h5')
    print("loaded")

    # a = str(input())

    with open('data/word_index' + dataset + '.json') as f:
        word_index = json.load(f)

    if a is not "":
        # print(word_index)
        x = get_encoded_matrix(dict(word_index), np.copy([a]), 250)
        # print(x)
        b = model.predict(x, batch_size=1, verbose=0)

        print(str((b[0][0] - 0.5) * 200) + "%", "is the sentiment")
        # print(b.shape)
    else:
        a, b = parse_kaggle()

        # print(word_index)
        x = get_encoded_matrix(dict(word_index), a, 250)
        # print(x)
        y = model.predict(x, batch_size=1, verbose=1)

        c = 0
        for i in range(len(y)):
            v = round(y[i][0])
            if v == b[i]:
                c = c + 1
            else:
                print(b[i], ", \"", a[i], "\"")

        print("Testing accuracy", c / len(y) * 100, "%")


def main(argv):
    global opts
    try:
        opts, args = getopt.getopt(argv, "ki:ho:sml")
    except getopt.GetoptError:
        print('test.py -[k(i <input>)] -[sml] -o model ')
        sys.exit()

    dataset = ""
    modelname = "-cnn"
    epochs = 8
    a = ""
    activation = ""

    for opt, arg in opts:
        if opt == '-i':
            a = str(arg)
        if opt == '-h':
            print('test.py -[k(i <input>)] -[sml] -o model ')
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
            activation = "/" + str(arg)
    # print("test", modelname[1:], "for", epochs, "epochs on", dataset[1:], "dataset")
    test(a, dataset=dataset, modelname=modelname, activation=activation)


if __name__ == "__main__":
    main(sys.argv[1:])
