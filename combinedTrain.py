from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, SimpleRNN, GRU, LSTM, Conv1D, GlobalMaxPooling1D, Dropout
import numpy as np

"""
Combined train is made to train all models for certain number of epochs with a certain batch size. train.py is the 
original true file. 
This file's purpose is only to make life easier. Since executing a file and leaving it like that is easier 
and more automated.
"""


def save_model(model_to_save, index=""):
    model_json = model_to_save.to_json()
    with open("data/model_to_save" + index + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model_to_save.save_weights("data/model_to_save" + index + ".h5")
    print("Saved model_to_save to disk")


epochs = 50
max_features = 20000
max_len = 250  # cut texts after this number of words (among top max_features most common words)
batch_size = 128

print('Loading data...')

# loads data from keras.datasets imdb
(x_train, y_train), (x_test, y_test) = (np.load("data/x_train.npy"), np.load("data/y_train.npy")) \
    , (np.load("data/x_val.npy"), np.load("data/y_val.npy"))
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
embedding_matrix = np.load("data/embedding_matrix.npy")
print("word vector dimension", len(embedding_matrix[0]))

# print(y_train)

# print('Build model RNN...')
#
#
# model = Sequential()
# model.add(Embedding(len(embedding_matrix), len(embedding_matrix[0]), weights=[embedding_matrix], trainable=False,
#                     input_length=max_len))
# model.add(SpatialDropout1D(0.2))
# model.add(SimpleRNN(256, dropout=0.2))
# model.add(Dense(1, activation='sigmoid'))
#
# # try using different optimizers and different optimizer configs
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# print('Train RNN...')
#
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_data=(x_test, y_test),
#           verbose=2)
# score, acc = model.evaluate(x_test, y_test,
#                             batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)
# save_model(model, index="-amazon-rnn")

print('Build model GRU...')

model = Sequential()
model.add(Embedding(len(embedding_matrix), len(embedding_matrix[0]), weights=[embedding_matrix], trainable=False,
                    input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(GRU(256, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train GRU...')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          verbose=2)
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
save_model(model, index="-amazon-gru")

print('Build model LSTM...')

model = Sequential()
model.add(Embedding(len(embedding_matrix), len(embedding_matrix[0]), weights=[embedding_matrix], trainable=False,
                    input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
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
save_model(model, index="-amazon-lstm")

print('Build CNN model...')

model = Sequential()
model.add(Embedding(len(embedding_matrix), len(embedding_matrix[0]), weights=[embedding_matrix], trainable=False,
                    input_length=max_len))
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

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=50,
          validation_data=(x_test, y_test),
          verbose=2)
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
save_model(model, index="-amazon-cnn")
