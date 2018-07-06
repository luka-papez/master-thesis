from __future__ import print_function

import numpy as np
import pickle
import os
import gc

import constants
import vocabulary
import model

from keras.models import load_model
import editdistance
from natsort import natsorted
from keras import backend as K

test_images_path = '../dataset_images/validation'
test_labels_path = '../dataset_labels/validation'
number_of_test_files = len(os.listdir(test_images_path))

trained_models_path = '../trained_models'

"""
# ==================================== EVALUATE MODEL ====================================
"""
from batch_generator import BatchGenerator

data_chunk_size = 1024
test_batches, x_test, y_test = next(BatchGenerator(test_images_path, test_labels_path).generate(data_chunk_size))

correspondence = []
edit_distances = []
for model_name in os.listdir(trained_models_path):#[m for m in natsorted(os.listdir(trained_models_path))[::-1] if m.count('-') == 2][int((len(os.listdir('.')) - 20) / 2):]:
    #model = load_model(os.path.join(trained_models_path, model_name))
    model = model.get_model('LSTM')
    model.load_weights(os.path.join(trained_models_path, model_name))

    print()
    print('-' * 50)
    print(model_name)
    print(test_batches[0])

    batch_correspondence = []
    batch_edit_distances = []
    printed_first = False
    data = zip(model.predict_classes(x_test, constants.BATCH_SIZE), y_test)

    for predicted_classes, y_true in data:
      words_pred = [vocabulary.vocabulary[c] for c in predicted_classes]
      words_true = [vocabulary.vocabulary[np.argmax(row)] for row in y_true]
      
      # the prediction is cut at the end in the truth because the token there can simply be replaced with the <end>
      # token to obtain the correct sequence until the end. pads are ignores
      words_pred = words_pred[:words_true.index('<end>') + 1 if '<end>' in words_true else -1]
      words_true = words_true[:words_true.index('<end>') + 1 if '<end>' in words_true else -1]
      
      batch_correspondence.append(len([1 for x, y in zip(words_pred, words_true) if x == y]))
      batch_edit_distances.append(editdistance.eval(words_pred, words_true))
      
      print(batch_correspondence[-1])

      if not printed_first: # print one sample for visualization
        #print(['CORRECT' if x == y else (x, y) for x, y in zip(words_pred, words_true) if y != '<pad>'])
        print(words_pred)
        print()
        print(words_true)
        print()
        printed_first = True

    print('acc: {}, avg_edit_dst: {}'.format((np.array(batch_correspondence) / constants.MAX_SEQ_LEN).mean(), np.array(batch_edit_distances).mean()))
    correspondence.append((model_name, batch_correspondence))
    edit_distances.append((model_name, batch_edit_distances))

    pickle.dump(correspondence, open('./correspondence-{}.pickle'.format(model_name), 'wb'))
    pickle.dump(edit_distances, open('./edit_distances-{}.pickle'.format(model_name), 'wb'))

    exit(1)
