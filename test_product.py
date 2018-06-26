import pandas as pd
import json

import numpy as np
import sys
from keras.models import model_from_json
from preProcessing import get_encoded_matrix


json_file = open('data' + '/model-amazon' + "-lstm" + "_500k" + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('data' + '/model-amazon' + "-lstm" + "_500k" + '.h5')
print("loaded")
with open('data/word_index' + "_500k" + '.json') as f:
        word_index = json.load(f)

y = []
with open('review_amazon.txt') as f1: 
	for line in f1: 
		x = get_encoded_matrix(dict(word_index), np.copy([line]), 250)
		b = model.predict(x, batch_size=1, verbose=0)
		y.append((b[0][0] - 0.5) * 200)

with open('review_flipkart.txt') as f2: 
	for line in f2: 
		x = get_encoded_matrix(dict(word_index), np.copy([line]), 250)
		b = model.predict(x, batch_size=1, verbose=0)
		y.append((b[0][0] - 0.5) * 200)

with open('review_snapdeal.txt') as f3: 
	for line in f3: 
		x = get_encoded_matrix(dict(word_index), np.copy([line]), 250)
		b = model.predict(x, batch_size=1, verbose=0)
		y.append((b[0][0] - 0.5) * 200)

data = pd.DataFrame({"percentage":y})
print(data.describe(include = 'all'))

