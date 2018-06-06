import json

import numpy as np
from keras.models import model_from_json

from cleanAmazon import get_encoded_matrix

json_file = open('data/model-amazon-gru.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('data/model-amazon-gru.h5')
print(model.summary())
print("loaded")

a = str(input())

with open('data/word_index.json') as f:
    word_index = json.load(f)

# print(word_index)
x = get_encoded_matrix(dict(word_index), np.copy([a]), 250)
# print(x)
b = model.predict(x, batch_size=1, verbose=0)

print(str((b[0][0] - 0.5) * 200)+"%", "is the sentiment")
# print(b.shape)
