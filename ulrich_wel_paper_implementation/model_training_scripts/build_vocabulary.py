from __future__ import print_function
import os
from tqdm import tqdm

import pickle
import constants

images_path = '../dataset_images'
labels_path = '../dataset_labels'

target_texts = []
target_words = set()
target_words.add('<pad>')

for root_dir, dirs, files in os.walk(images_path):
  for i, img_path in tqdm(enumerate(files)):
    if os.path.exists(os.path.join(root_dir, img_path)):
      label_path = os.path.join(root_dir.replace('images', 'labels'), img_path[:-6] + '.labels')

      if not os.path.exists(label_path):
        print('Skipping', label_path)
        continue
        
      with open(label_path, 'r') as lf:
        # - 2 leaves some space for begin and end tokens
        target_text = lf.read().strip().split(' ')[:constants.MAX_SEQ_LEN - 2]
        number_of_tokens = len(target_text)
        target_text = ['<begin>'] + target_text + ['<end>']
        target_texts.append(target_text)

        for word in target_text:
            if word not in target_words:
                target_words.add(word)

target_words = sorted(list(target_words))
num_decoder_tokens = len(target_words)
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(target_texts))
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for outputs:', max_decoder_seq_length)

target_token_index = dict([(word, i) for i, word in enumerate(target_words)])

with open('texts.pickle', 'wb') as f:
  pickle.dump(target_texts, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('dictionary.pickle', 'wb') as f:
  pickle.dump(target_token_index, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('vocabulary.pickle', 'wb') as f:
  pickle.dump(target_words, f, protocol=pickle.HIGHEST_PROTOCOL)
