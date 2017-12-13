'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.layers import Flatten
max_words = 1000
batch_size = 32
epochs = 5

print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
#x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
#x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
#x_train = tokenizer.sequences_to_matrix(x_train, mode='count')
#x_test = tokenizer.sequences_to_matrix(x_test, mode='count')

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train[0])
print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
max_review_length=200
x_train=sequence.pad_sequences(x_train,maxlen=max_review_length)
x_test=sequence.pad_sequences(x_test,maxlen=max_review_length)
print('Building model...')
#num_classes = np.max(y_train) + 1
#x_train=x_train[:,:,np.newaxis];
#x_test=x_test[:,:,np.newaxis];
model=Sequential()
model.add(Embedding(1000, 32, input_length=200))
model.add(Conv1D(250,3,padding='same',activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
print("num_clasees")
print(num_classes)
model.add(Dense(int(num_classes)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
