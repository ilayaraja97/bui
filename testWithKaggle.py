import json

import numpy as np
from keras.models import model_from_json

from preProcessing import get_encoded_matrix
from parseKaggle import parse_kaggle

model_name = "rnn"

json_file = open('data/model-amazon-' + model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('data/model-amazon-' + model_name + '.h5')
# print(model.summary())
print("loaded")

a, b = parse_kaggle()

with open('data/word_index.json') as f:
    word_index = json.load(f)

# print(word_index)
x = get_encoded_matrix(dict(word_index), a, 250)
# print(x)
y = model.predict(x, batch_size=1, verbose=1)

c = 0
for i in range(len(y)):
    v = round(y[i][0])
    if v == b[i]:
        c = c + 1

print("Testing accuracy", c/len(y)*100, "%")

# print(b.shape)
