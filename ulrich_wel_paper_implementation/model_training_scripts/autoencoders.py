from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras import backend as K

import constants

input_img = Input(shape=(constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 1))

filters = [64, 32, 16, 8]

x = Conv2D(filters[0], (3, 3), padding='same')(input_img)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(filters[1], (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(filters[2], (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(filters[3], (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# decoder
x = Conv2D(filters[3], (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(filters[2],(3, 3), padding='same')(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(filters[1], (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(filters[0], (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(1, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

print(autoencoder.summary())

required_memory = constants.get_model_memory_usage(constants.BATCH_SIZE, autoencoder)
print('Space required for model training: {} gigabytes'.format(required_memory))

if required_memory >= 5.0:
  print('Not enough memory to train on laptop')
  #exit(1)

####################################################################

import numpy as np
import os
import cv2

X = None
for root_dir, dirs, files in os.walk('../dataset_images/train'):
  X = np.zeros(shape=(len(files[0:2000]), constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH))

  for i, s in enumerate(files[0:2000]):
    if i % 1000 == 0:
      print('loaded files: ' + str(i))

    curr_img = cv2.bitwise_not(cv2.imread(os.path.join(root_dir, s), cv2.IMREAD_UNCHANGED)[:, :, 3])
    X[i] = curr_img

    #denoised_filepath = os.path.join('../dataset_images/train', key_row)
    #Y[i] = cv2.bitwise_not(cv2.imread(denoised_filepath, cv2.IMREAD_UNCHANGED)[:, :, 3])

x_train = X[ : 7 * len(X) / 10].astype('float32') / 255.
x_val = X[7 * len(X) / 10 : 9 * len(X) / 10].astype('float32') / 255.
x_test =  X[9 * len(X) / 10 : ].astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 1))
x_val = np.reshape(x_val, (len(x_val), constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 1))
x_test = np.reshape(x_test, (len(x_test), constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 1))

y_train = x_train
y_val = x_val
y_test = x_test

print(x_test.any())

####################################################################

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

import os.path
model_path = 'my_model.hdf5'
filepath = '../trained_models/weights.{epoch:02d}-{val_loss:.2f}.hdf5'

import os
os.system('free -m')

if not os.path.exists(model_path):
  autoencoder.fit(x_train, y_train,
                  epochs=30,
                  batch_size=constants.BATCH_SIZE,
                  shuffle=True,
                  validation_data=(x_val, y_val),
                  callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), 
                             ModelCheckpoint(filepath)])
  autoencoder.save(model_path)

else:
  from keras.models import load_model
  autoencoder = load_model(model_path)

####################################################################    

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(8, 64, 8))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

print(encoded_imgs.shape)
print(decoded_imgs.shape)

decoded_imgs = autoencoder.predict(x_test)

for i, pred in enumerate(decoded_imgs):
  img = (np.concatenate((x_test[i], pred, y_test[i]), axis=0) * 255.0).astype('uint8')
  cv2.imwrite('autoencoder/{}.png'.format(i), img)
  #cv2.namedWindow('result')
  #cv2.imshow('result', img)
  #cv2.waitKey(0)


