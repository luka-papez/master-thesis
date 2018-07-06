import numpy as np
import cv2
import pickle
import os
from math import ceil

import constants
import vocabulary

class BatchGenerator(object):

  def __init__(self, images_path, labels_path):
    self.images_path = images_path
    self.labels_path = labels_path

  def generate(self, data_chunk_size=128):
    for root_dir, dirs, files in os.walk(self.images_path):
      if not files:
        continue

      while True:
        dataset_size = len(files)

        for b in range(0, dataset_size, data_chunk_size):
          batch = files[b : b + data_chunk_size]
          batch_len = len(batch)

          X = np.zeros(shape=(batch_len, constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH), dtype=np.float32)
          Y = np.zeros(shape=(batch_len, vocabulary.max_decoder_seq_length, vocabulary.vocabulary_size), dtype=np.float32)

          i = 0
          while i < batch_len:
              for img_path in batch:
                if i % (data_chunk_size / 4) == 0:
                  pass #print('Loaded {} files'.format(i))

                if i == batch_len:
                    break

                label_path = os.path.join(self.labels_path, img_path[:-6] + '.labels')
                if not os.path.exists(label_path):
                  #print(str(ValueError('Cannot find labels for ' + label_path)))
                  continue

                with open(label_path, 'r') as f:
                  target_text = f.read().strip().split(' ')[:constants.MAX_SEQ_LEN - 2]
                  target_text = ['<begin>'] + target_text + ['<end>']
                  target_text *= int(ceil(constants.MAX_SEQ_LEN / len(target_text)))
                  target_text = target_text[:constants.MAX_SEQ_LEN]
                  #target_text = ['<begin>'] + target_text + ['<end>'] + ['<pad>'] * (constants.MAX_SEQ_LEN - (2 + target_text_len))

                  for t, word in enumerate(target_text):
                    Y[i, t, vocabulary.dictionary[word]] = 1.0

                  curr_img = cv2.imread(os.path.join(root_dir, img_path), cv2.IMREAD_UNCHANGED)
                  X[i] = curr_img[:, :, -1].astype('float32') / 255.

                  i = i + 1

          yield batch, np.reshape(X, (batch_len, constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 1)),\
                       np.reshape(Y, (batch_len, vocabulary.max_decoder_seq_length, vocabulary.vocabulary_size))
