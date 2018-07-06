"""
  An example script for dataset download from 
    OPTICAL MUSIC RECOGNITION WITH CONVOLUTIONAL SEQUENCE-TO-SEQUENCE MODELS
    by Eelco van der Wel and Karen Ullrich
  modified for master thesis to download dataset files while checking for already downloaded files.
  The script can be terminated and resumed at will, the download will simply continue at the next downloaded file.
  A MuseScore API key is required to download the files and must be placed inside "api_key.py" module.
  
  Author: Luka Papež
  Date: Spring 2018.
"""

'''
An example of how to use the MuseScore API
with the MuseScore Monophonic MusicXML dataset.

Author: Eelco van der Wel
Date: April 2017
'''

import urllib.request
import json
import os
import argparse


class ProgressLoggingIterable(object):
    def __init__(self, values):
        self.values = list(values)
        self.location = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.location == len(self.values):
            raise StopIteration
        value = self.values[self.location]
        self.location += 1
        print('Completed: {} of {}.'.format(self.location, len(self.values)))
        return value
        
    def next(self):
      return self.__next__()


def get_progress_logging_iterable(iterable):
  try:
    from tqdm import tqdm
  except ImportError as e:
    print('WARNING: tqdm is not installed, please install it for prettier download display')
    return ProgressLoggingIterable(iterable)
  else:
    return tqdm(iterable)


def determine_missing_files(key_list, download_dir):
  dataset_scores = []
  with open(key_list, 'r') as f:
      dataset_scores = set(f.read().splitlines())
  
  already_have = set()
  for root_dir, dirs, files in os.walk(download_dir):
    for f in files:
      current_file = os.path.join(root_dir, f)  
      filename, file_extension = os.path.splitext(current_file)
      filename = os.path.basename(filename)
      already_have.add(filename)
  
  print('Already have {} scores'.format(len(already_have)))
  return dataset_scores - already_have
  

if __name__ == '__main__':
  try:
    # needed to download scores, contact musescore.org to obtain the key
    from api_key import api_key 
  except ImportError as e:
    print('Define your API key first in api_key.py.')
    raise e
    
  parser = argparse.ArgumentParser(prog=os.sys.argv[0], usage='python3 %(prog)s [options]')
  parser.add_argument('--keys', '-k', help='A file containing score keys from Musescore API that will be downloaded')
  parser.add_argument('--dest', '-d', help='Download destination folder')
  
  if len(os.sys.argv) < 2:
    parser.print_help()
    exit(1)
  
  args = parser.parse_args()
  print(args)
  
  # We need to download 2 components: 
  # - a json with the score information, containing the secret id
  # - the score mxl file, using the public and secret id
  score_json_url = 'http://api.musescore.com/services/rest/score/{}.json?oauth_consumer_key='
  score_json_url += api_key
  score_file_url = 'http://static.musescore.com/{}/{}/score.mxl'

  to_download = determine_missing_files(args.keys, args.dest)  
  for score_id in get_progress_logging_iterable(to_download):
    # First download score JSON to get secret
    r = None
    try:
      r = urllib.request.urlopen(score_json_url.format(score_id))
    except Exception as e:
      print(score_id + ' ' + str(e))
      continue
        
    score_json = json.loads(r.read().decode(r.info().get_param('charset') or 'utf-8'))
    score_secret = score_json['secret']

    # Define save location
    filename = os.path.join(args.dest, '{}.mxl'.format(score_id))
    print (filename)
    # Download score
    try:
      urllib.request.urlretrieve(score_file_url.format(score_id, score_secret), filename)
    except Exception as e:
      print(score_id + ' ' + str(e))
      continue

  print('Done!')
