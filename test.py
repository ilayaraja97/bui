import numpy as np
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.preprocessing.text import  text_to_word_sequence

json_file = open('data/modelimdb-lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('data/modelimdb-lstm.h5')
print("loaded")

a = "raja sexy" # str(input())
x = text_to_word_sequence(a)
print(x)
x = sequence.pad_sequences(x, maxlen=80)
print(x)
b = model.predict(x, batch_size=1, verbose=1)

print(b)
print(b.shape)
