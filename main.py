from models import getData, getEmbedding
from splitter import splitData



data = getData()
word_index, data, embedding_matrix = getEmbedding(data)
x_train, y_train, x_val, y_val = splitData(data, split_value=0.3)