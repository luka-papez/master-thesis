import os
from tqdm import tqdm

USELESS_TOKENS = frozenset(['TODO:', 'duration', 'quintuplet', '128th', '0', '-1', '8', 'breve', 'long', 'quarter-sharp', '9', 'quarter-flat', '256th', 'three-quarters-sharp', 'natural-down', 'flat-down', 'slash-sharp'])
print(len(USELESS_TOKENS), USELESS_TOKENS)

for root_dir, dirs, files in os.walk('../dataset_labels_48_classes'):
    for file_name in tqdm(files):
        folder = root_dir.split('\\')[1]
        with open(os.path.join(root_dir, file_name), 'r') as input_file:
            tokens = input_file.read().strip().split(' ')
            fixed_tokens = ['Other' if t in USELESS_TOKENS else t for t in tokens]
            with open(os.path.join('../dataset_labels', folder, file_name), 'w') as output_file:
                output_file.write(' '.join(fixed_tokens))
