""" 
  This script applies various transformations described in the paper
  OPTICAL MUSIC RECOGNITION WITH CONVOLUTIONAL SEQUENCE-TO-SEQUENCE MODELS
  by Eelco van der Wel and Karen Ullrich
  to fragments of sheet music (each staff is in a separate image file)
  
  Author: Luka Papez
  Date: Spring 2018
"""

import os
import cv2
import numpy as np
import argparse
import noise
from tqdm import tqdm

def apply_additive_noise_map(image, noise):  
  image = image.astype('int32')
  noise = noise.astype('int32')
  
  image += noise
  
  image = np.clip(image, 0, 255)
  image = image.astype('uint8')

  return image


def additive_gaussian_white_noise(image):
  # TODO: args
  # TODO: average on the whole dataset as mu as is described in the paper
  mu = 10
  sigma = 5
  gauss = np.random.normal(mu, sigma, size=image.shape)
  
  return apply_additive_noise_map(image, gauss)
  

def additive_perlin_noise(image):
  # https://stackoverflow.com/questions/45686963/perlin-noise-looks-too-griddy

  # TODO: args
  intensity = 0.8
  image_height, image_width = image.shape
  freq = image_height
  image_randomness = np.random.rand(1)
  
  freq += np.random.rand(1) * (image_height / 10)

  perlin = np.empty(shape=image.shape, dtype=np.float32)

  for y in range(image_height):
    for x in range(image_width):
      # using 3D noise to allow for different noise in each image by putting a random z coordinate
      perlin[y][x] = noise.pnoise3(float(x) / freq, float(y) / freq, image_randomness, octaves=4)
  
  # normalization
  max_val = np.max(perlin)
  min_val = np.min(perlin)
  span = max_val - min_val
  perlin = ((perlin - min_val) / span) * 255 * intensity
  perlin = perlin.astype('uint8')
    
  return apply_additive_noise_map(image, perlin)


"""
  Elastic deformation of images as described in [Simard2003] Simard, Steinkraus and Platt, 
  "Best Practices for Convolutional Neural Networks applied to Visual Document Analysis", 
  in Proc. of the International Conference on Document Analysis and Recognition, 2003.
  
  Taken from: https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
  1) fixed bugs with inverted coordinates
  2) adapted to work with music scores by adding the option to reduce the vertical component of the algorithm.
"""
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
def elastic_transformation(image, alpha, sigma, horizontal_intensity=1, vertical_intensity=1,random_state=None):
  """
  """
  assert len(image.shape)==2

  if random_state is None:
    random_state = np.random.RandomState(None)

  shape = image.shape

  dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha 
  dx *= horizontal_intensity
  dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha 
  dy *= vertical_intensity

  y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
  indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
  
  return map_coordinates(image, indices, order=1).reshape(shape)


def small_elastic_transformations(image, test_set):
  """
    alpha is chosen to be a random value between 2 and 8, 
    with a sigma of 0.5 for the training data, and a sigma of 2 for the evaluation data.
  """
  alpha = np.random.rand(1) * 6 + 2
  
  # NOTE: the value 0.5 used in the paper by Ulrich and Wel is simply too low 
  # it sometimes produces scores which cannot be interpreted even by a human and 
  # there is no sense in learning the model on such data (which is basically just noise)
  # 1.5 works okay
  sigma = 2 if test_set else 1.5
  
  return elastic_transformation(image, alpha, sigma)

  
def large_elastic_transformations(image, test_set):
  """
    An alpha between 2000 and 3000 is used, with a sigma of 40 for the training data, and 80 for the evaluation data. 
    To maintain straight and continuous stafflines, the original algorithm is slightly adapted to reduce vertical
    translations of pixels by reducing the vertical component of transformations by 95%.
  """
  alpha = np.random.rand(1) * 1000 + 2000
  sigma = 80 if test_set else 40
  
  # NOTE: the 95% reduction in vertical component used in the paper by Ulrich and Wel is a bit extreme
  # it might be beneficial for the model to learn vertical deformations 
  # (i.e. folds in the paper or curvature of the book)
  # 75% reduction makes the deformation a bit more realistic
  return elastic_transformation(image, alpha, sigma, 1, 0.25)


def debug_display_image(image, file_name):
  cv2.imshow(file_name, cv2.bitwise_not(image_noised))
  key = cv2.waitKey(0) & 0xff
  cv2.destroyWindow(file_name)
  
  # 'ESC' to exit the program, any other key just continues
  if key == 27:
    exit(1)


def main(args):
  source_directory = args.src
  dest_directory = args.dst

  newly_created = set()
  for root_dir, dirs, files in list(os.walk(source_directory)):
    for f in tqdm(files):
      current_file = os.path.join(root_dir, f)  
      filename, file_extension = os.path.splitext(current_file)
      filename = os.path.basename(filename)
      
      if filename not in newly_created and '-1' in filename:
        filename = filename.replace('-1', '')
      else:
        continue
      
      image = cv2.imread(current_file, cv2.IMREAD_UNCHANGED)
      
      # take just the alpha channel as the image (this immediately converts it to grayscale)
      # because MuseScore outputs the image as .png with everything zeroes except the alpha channel
      image = image[:, :, 3]

      for i in range(int(args.number_of_images)):
        image_noised = np.copy(image)

        image_noised = small_elastic_transformations(image_noised, args.mode == 'test')
        image_noised = large_elastic_transformations(image_noised, args.mode == 'test')
        image_noised = additive_gaussian_white_noise(image_noised)
        image_noised = additive_perlin_noise(image_noised)
      
        # worked with inverted images so far
        image_noised = cv2.bitwise_not(image_noised)
        new_name = filename + '_augm_{}'.format(i)
        newly_created.add(new_name)
        cv2.imwrite(os.path.join(dest_directory, new_name + '.png'), image_noised)
        
        # debug_display_image(image_noised, current_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog=os.sys.argv[0], usage='python %(prog)s [options]')
  parser.add_argument('--src', '-s', help='Folder containing images to process')
  parser.add_argument('--dst', '-d', help='Folder where to put the resulting images')
  parser.add_argument('--number_of_images', '-n', help='How many new images to produces', default=10)
  parser.add_argument('--mode', '-e', help='Defines the mode of the dataset being created (test or train)', default='train')
  
  main(parser.parse_args())
  
