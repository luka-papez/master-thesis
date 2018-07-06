from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.layers import Flatten, RepeatVector, TimeDistributed, Reshape, Bidirectional, Lambda
from keras.layers import LSTM, RNN, CuDNNGRU, CuDNNLSTM
from keras.models import Model, Sequential
from keras import backend as K

import constants
import vocabulary

def create_lstm_simple_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((4, 4), padding='same'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((4, 4), padding='same'))

    model.add(Flatten())
    model.add(RepeatVector(vocabulary.max_decoder_seq_length))

    model.add(CuDNNLSTM(vocabulary.vocabulary_size, return_sequences=True))

    model.add(Lambda(lambda x: K.tf.nn.softmax(x, axis=1)))

    return model

def create_lstm_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Flatten())
    model.add(RepeatVector(vocabulary.max_decoder_seq_length))

    #model.add(Bidirectional(CuDNNLSTM(constants.LSTM_HIDDEN_SIZE, return_sequences=True)))
    model.add(Bidirectional(LSTM(constants.LSTM_HIDDEN_SIZE, return_sequences=True)))    
    model.add(Dense(vocabulary.vocabulary_size))

    model.add(Lambda(lambda x: K.tf.nn.softmax(x, axis=1)))

    return model

def create_fc_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Flatten())

    model.add(Dense(int(vocabulary.vocabulary_size * vocabulary.max_decoder_seq_length)))
    model.add(Reshape((vocabulary.max_decoder_seq_length, vocabulary.vocabulary_size)))

    model.add(Lambda(lambda x: K.tf.nn.softmax(x, axis=1)))

    return model

def create_conv_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Reshape((vocabulary.max_decoder_seq_length, vocabulary.vocabulary_size)))
    model.add(Lambda(lambda x: K.tf.nn.softmax(x, axis=1)))

    return model

def create_ulrich_wel_model():
    model = Sequential()

    model.add(MaxPooling2D((3, 3), padding='same', input_shape=(constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 1)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(MaxPooling2D((3, 3), padding='same'))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Flatten())
    model.add(RepeatVector(vocabulary.max_decoder_seq_length))

    model.add(Bidirectional(CuDNNLSTM(constants.LSTM_HIDDEN_SIZE, return_sequences=True)))
    model.add(Dense(vocabulary.vocabulary_size))

    model.add(Lambda(lambda x: K.tf.nn.softmax(x, axis=1)))

    return model

def create_perceptron_model():
    model = Sequential()

    model.add(MaxPooling2D((4, 4), padding='same', input_shape=(constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 1)))
    model.add(Flatten())

    model.add(Dense(vocabulary.max_decoder_seq_length * vocabulary.vocabulary_size, activation='relu'))
    model.add(Reshape((vocabulary.max_decoder_seq_length, vocabulary.vocabulary_size)))
    model.add(Lambda(lambda x: K.tf.nn.softmax(x, axis=1)))

    return model

class NothingClassifier(object):
    def predict_classes(self, x_test, *args, **kwargs):
        import numpy as np
        return np.array(len(x_test) * [[vocabulary.dictionary['<pad>']] * vocabulary.max_decoder_seq_length])

class StatisticalClassifier(object):
    def predict_classes(self, x_test, *args, **kwargs):
        import numpy as np
        tokens = ['<begin>'] + ['C', '4', 'eighth'] * 24 + ['<end>']
        tokens += ['<pad>'] * (128 - len(tokens))
        tokens = [list(map(lambda x: vocabulary.dictionary[x], tokens))]
        tokens *= len(x_test)
        return np.array(tokens)

def get_model(model_name):
  model_name = model_name.lower()

  if 'lstm' in model_name:
    return create_lstm_model()
  elif 'gru' in model_name:
    return create_gru_model()
  elif 'fully' in model_name:
    return create_fc_model()
  elif 'conv' in model_name:
    return create_conv_model()
  elif 'nothing' in model_name:
    return NothingClassifier()
  elif 'stats' in model_name:
    return StatisticalClassifier()
  elif 'simple' in model_name:
    return create_lstm_simple_model()
  elif 'ulrich' in model_name or 'wel' in model_name:
    return create_ulrich_wel_model()
  elif 'perceptron' in model_name:
    return create_perceptron_model()
  else:
    return None
