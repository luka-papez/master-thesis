import pickle

with open('dictionary.pickle', 'rb') as f:
  dictionary = pickle.load(f)

with open('vocabulary.pickle', 'rb') as f:
  vocabulary = pickle.load(f)

with open('texts.pickle', 'rb') as f:
  texts = pickle.load(f)

max_decoder_seq_length = max([len(txt) for txt in texts])
vocabulary_size = len(vocabulary)

print('=' * 50)
print('Maximum sequence length: {}'.format(max_decoder_seq_length))
print('=' * 50)
print('Vocabulary size: {}'.format(vocabulary_size))
print('=' * 50)
lengths = [len(text) for text in texts]
print('Average number of tokens in text: {}'.format(float(sum(lengths)) / len(lengths)))
print('=' * 50)

del texts # saves memory
