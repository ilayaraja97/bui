from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, SimpleRNN, GRU, LSTM
import numpy as np


def save_model(model, index=""):
    model_json = model.to_json()
    with open("data/model" + index + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("data/model" + index + ".h5")
    print("Saved model to disk")


epochs = 50
max_features = 20000
max_len = 250  # cut texts after this number of words (among top max_features most common words)
batch_size = 128

print('Loading data...')

# loads data from keras.datasets imdb
(x_train, y_train), (x_test, y_test) = (np.load("data/x_train.npy"), np.load("data/y_train.npy")) \
    , (np.load("data/x_val.npy"), np.load("data/y_val.npy"))
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')
# print(x_train[1])
# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, max_len=max_len)
# x_test = sequence.pad_sequences(x_test, max_len=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
embedding_matrix = np.load("data/embedding_matrix.npy")
print("word vector dimension", len(embedding_matrix[0]))
#print(y_train)

print('Build model RNN...')


model = Sequential()
model.add(Embedding(len(embedding_matrix), len(embedding_matrix[0]), weights=[embedding_matrix], trainable=False,
                    input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(SimpleRNN(256, dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train RNN...')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          verbose=2)
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
save_model(model, index="-amazon-rnn")

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
