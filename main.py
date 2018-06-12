import numpy as np
import json

from preProcessing import splitData, getEmbedding, get_encoded_matrix
from parseAmazon import parse_amazon
from parseKaggle import parse_kaggle

x, y = parse_amazon()
word_index, data, embedding_matrix = getEmbedding((x, y))

# validate with kaggle
# x1, y1 = parse_kaggle()
# x1 = get_encoded_matrix(dict(word_index), x1, 250)
# word_index2, data2, embedding_matrix2 = getEmbedding((x1, y1))
# x_train, y_train, x_val, y_val = data[0], data[1], x1, y1

# validate with amazon
x_train, y_train, x_val, y_val = splitData(data, split_value=0.1)

print(x_train)
print(y_train)
print(x_val)
print(y_val)
with open('data/word_index.json', "w") as outf:
    json.dump(word_index, outf)
np.save('data/embedding_matrix', embedding_matrix)
np.save('data/x_train', x_train)
np.save('data/y_train', y_train)
np.save('data/x_val', x_val)
np.save('data/y_val', y_val)
