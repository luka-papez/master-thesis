import os
from batch_generator import BatchGenerator
import model
import constants
import vocabulary
import numpy as np
import editdistance
from tqdm import tqdm

test_images_path = '../dataset_images/test'
test_labels_path = '../dataset_labels/test'

model_name = 'STATS'
#model_name = 'nothing'
model = model.get_model(model_name)

data_chunk_size = 1024
test_batches, x_test, y_test = next(BatchGenerator(test_images_path, test_labels_path).generate(data_chunk_size))

correspondence = []
edit_distances = []

batch_correspondence = []
batch_edit_distances = []
y_pred = model.predict_classes(x_test, batch_size=constants.BATCH_SIZE)

printed_first = False
for predicted, true in tqdm(zip(y_pred, y_test)):
  words_pred = [vocabulary.vocabulary[c] for c in predicted]
  words_true = [vocabulary.vocabulary[np.argmax(row)] for row in true]

  batch_correspondence.append(len([1 for x, y in zip(words_pred, words_true) if x == y]))
  batch_edit_distances.append(editdistance.eval(words_pred, words_true))

  if not printed_first: # print one sample for visualization
    print(['CORRECT' if x == y else (x, y) for x, y in zip(words_pred, words_true) if y != '<pad>'])
    print()
    printed_first = True

print('acc: {}, avg_edit_dst: {}'.format((np.array(batch_correspondence) / constants.MAX_SEQ_LEN).mean(), np.array(batch_edit_distances).mean()))
correspondence.append((model_name, batch_correspondence))
edit_distances.append((model_name, batch_edit_distances))
