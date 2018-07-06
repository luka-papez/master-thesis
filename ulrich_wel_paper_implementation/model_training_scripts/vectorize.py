from __future__ import print_function

from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint

import numpy as np
import cv2
import pickle

import os
from tqdm import tqdm

import constants
import vocabulary
import model

train_images_path = '../dataset_images/train'
train_labels_path = '../dataset_labels/train'
valid_images_path = '../dataset_images/validation'
valid_labels_path = '../dataset_labels/validation'

number_of_train_files = len(os.listdir(train_images_path))
number_of_valid_files = len(os.listdir(valid_images_path))

MODEL_NAME = 'LSTM'
#MODEL_NAME = 'FULLY_CONN'
#MODEL_NAME = 'CONV'
#MODEL_NAME = 'SIMPLE'
#MODEL_NAME = 'WEL'
#MODEL_NAME = 'PERCEPTRON'

model = model.get_model(MODEL_NAME)

print(model.summary())

required_memory = constants.get_model_memory_usage(constants.BATCH_SIZE, model)
print('Space required for model training: {} gigabytes'.format(required_memory))

if required_memory >= 1.5:
  print('Not enough memory to train on laptop')
  #exit(1)

"""
# ==================================== TRAIN MODEL ====================================
"""
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy'])

from batch_generator import BatchGenerator
data_chunk_size = 1024
dataset = BatchGenerator(train_images_path, train_labels_path).generate(data_chunk_size)
val_batch, x_val, y_val = next(BatchGenerator(valid_images_path, valid_labels_path).generate(data_chunk_size))
val_example, x_exa, y_exa = next(BatchGenerator(valid_images_path, valid_labels_path).generate(constants.BATCH_SIZE))

history_folder = '../history'
models_folder = '../trained_models'
for iteration in range(1, 20 + 1):
    print()
    print('-' * 50)

    chunk = 1
    for batch_files, x_train, y_train in dataset:
        train_history = model.fit(x_train, y_train, batch_size=constants.BATCH_SIZE, epochs=1, verbose=0)

        scores = model.evaluate(x_val, y_val, verbose=0, batch_size=constants.BATCH_SIZE)
        print("Iteration %d, chunk %d, (%d/%d) | train_loss: %.2f, train_acc: %.2f%%, val_acc: %.2f%%" % \
          (iteration, \
           chunk, \
           chunk * data_chunk_size, \
           number_of_train_files, \
           train_history.history['loss'][0], \
           train_history.history['categorical_accuracy'][0] * 100, \
           scores[1] * 100 \
           ))

        for p, (classes, y_true) in enumerate(zip(model.predict_classes(x_exa, batch_size=constants.BATCH_SIZE), y_exa)):
          words_pred = [vocabulary.vocabulary[c] for c in classes]
          words_true = [vocabulary.vocabulary[np.argmax(row)] for row in y_true]

          if p == 0:
            print(['CORRECT' if x == y else (x, y) for x, y in zip(words_pred, words_true) if y != '<pad>'])
            print()

        # a single point of history for the graph
        # (train_loss, train_acc, val_loss, val_acc)
        history_point = (train_history.history['loss'][0], train_history.history['categorical_accuracy'][0], scores[0], scores[1])
        with open(os.path.join(history_folder, 'history-{}-{}-{}.pickle'.format(MODEL_NAME, iteration, chunk)), 'wb') as f:
            pickle.dump(history_point, f)

        if data_chunk_size > 1000:
            model.save(os.path.join(models_folder, 'model-{}-{}-{}.h5'.format(MODEL_NAME, iteration, chunk)))

        chunk = chunk + 1
        if chunk > number_of_train_files / data_chunk_size:
          break

    model.save(os.path.join(models_folder, 'model-{}-{}.h5'.format(MODEL_NAME, iteration)))
