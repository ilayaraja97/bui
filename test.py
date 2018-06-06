import json
import re

import numpy as np
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence

from models import getEmbedding


def getIdMatrixFromSentences(vocab, data, maxSeqLength):
    ids = np.zeros([len(data), maxSeqLength], dtype=int)
    for i, sentence in enumerate(data):
        for j, word in enumerate(re.split("[ !\"#$%&*+,-./:;<=>?@^_`{|}~\t\n']", sentence)):
            if j == maxSeqLength:
                break
            if word.lower() in vocab:
                # print(word)
                ids[i][j] = vocab.get(word.lower())
            else:
                ids[i][j] = 0
    return ids


json_file = open('data/model-amazon-gru.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('data/model-amazon-gru.h5')
print("loaded")

a = str(input())

with open('data/word_index.json') as f:
    word_index = json.load(f)

# print(word_index)
x = getIdMatrixFromSentences(dict(word_index), np.copy([a]), 250)
# print(x)
b = model.predict(x, batch_size=1, verbose=0)

print(str((b[0][0] - 0.5) * 200)+"%", "is the sentiment")
# print(b.shape)
