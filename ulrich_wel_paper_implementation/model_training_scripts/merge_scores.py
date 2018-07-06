from __future__ import print_function

import numpy as np
import os

import constants
import vocabulary
import model

from keras.models import load_model
import editdistance
from natsort import natsorted
from keras import Model, backend as K
from keras.layers import Input

test_images_path = '../dataset_images/validation'
test_labels_path = '../dataset_labels/validation'
number_of_test_files = len(os.listdir(test_images_path))

trained_models_path = '../trained_models'

from batch_generator import BatchGenerator

data_chunk_size = 1024
test_batches, x_test, y_test = next(BatchGenerator(test_images_path, test_labels_path).generate(data_chunk_size))

for model_name in os.listdir(trained_models_path):
    model = model.get_model('LSTM')
    model.load_weights(os.path.join(trained_models_path, model_name))

    print(model_name)
    print(test_batches[0])
    print(test_batches[1])

    get_encoder_output = K.function([model.input], [model.get_layer('flatten_layer').output])
    get_decoder_output = K.function([model.get_layer('flatten_layer').output], [model.output])

    encoder_output = get_encoder_output([x_test])[0]
    joined_scores = np.array([(encoder_output[0] + encoder_output[1]) / 2], dtype=encoder_output.dtype)
    decoder_output = get_decoder_output([joined_scores])[0]

    for predicted_classes, y_true in zip(decoder_output, y_test):
      words_pred = [vocabulary.vocabulary[np.argmax(c)].encode('utf-8') for c in predicted_classes]
     
      print(words_pred)
      
    exit(1)
