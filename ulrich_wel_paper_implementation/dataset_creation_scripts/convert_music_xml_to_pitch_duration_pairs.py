"""
  This script converts notes contained in MusicXML files to simple pairs (pitch, duration) 
  which can be used as labels for a machine learning model.
  
  NOTE: well into this project I realized something like this already exists:
    https://github.com/tensorflow/magenta/blob/master/magenta/music/musicxml_parser.py
    https://github.com/tensorflow/magenta/blob/master/magenta/music/musicxml_reader.py
    https://github.com/tensorflow/magenta/blob/master/magenta/protobuf/music.proto
  
  Author: Luka Papez
  Date: Spring 2018
"""

import os
import sys
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm


def process_note(divisions_per_quarter_note, note):
  note_pitch = ''
  note_duration = ''
  
  def add_word(word, target):
    # in case the formatting is changed
    target += word + ' '
    
    return target
  
  # pitch
  if note.find('.//rest') is None:
    note_pitch = add_word(note.find('.//step').text, note_pitch)
    note_pitch = add_word(note.find('.//octave').text, note_pitch)
  
    alter = note.find('.//alter')
    if alter is not None:
      alter = float(alter.text)
      
      if abs(alter) == 2.0:
        #note_pitch = add_word('double', note_pitch)
        
        # TODO: this could be supported but it's rare so:
        return None

      note_pitch = add_word('flat' if alter < 0 else 'sharp', note_pitch)
    
    accidental = note.find('.//accidental')
    if accidental is not None:
      accidental = accidental.text
      
      note_pitch = add_word(accidental, note_pitch)
    
  else:
     note_pitch = add_word('rest', note_pitch)
     
  # duration
  # it's possible type is not always there, and duration is also not necessary
  # if both are missing well then everything will crash as it should
  # https://usermanuals.musicxml.com/MusicXML/Content/EL-MusicXML-duration.htm
  note_type = note.find('.//type')
  #if note_type is None:
  if note_type is None:
    duration = note.find('.//duration')
    
    if duration is None:
      raise ValueError('Note type not given and no note duration found')
      
    # hacky, not gonna explain this part just accept it
    duration = int(duration.text)
    duration_in_parts = 4 * divisions_per_quarter_note / duration
   
    duration_names = { 
      0: 'brave', 1: 'whole', 2: 'half', 4: 'quarter', 8: 'eighth', 
      16: '16th', 32: '32nd', 64: '64th', 128: '128th', 256: '256th', 512: '512th', 1024: '1024th' 
    }
    
    if duration_in_parts in duration_names and duration % (4 * divisions_per_quarter_note) != 0:
      note_duration = add_word(duration_names[duration_in_parts], note_duration)
    else:
      # TODO: dotted and triplets based on calculation
      note_duration = add_word('TODO: duration', note_duration)
      
  else:    
    note_duration = add_word(note_type.text, note_duration)

    if note.find('.//dot') is not None:
      note_duration = add_word('dotted', note_duration)
    
  actual_notes = note.find('.//actual-notes')
  if actual_notes is not None:
    actual_notes = int(actual_notes.text)
    if actual_notes == 3:
      note_duration = add_word('triplet', note_duration)
    elif actual_notes == 5:
      note_duration = add_word('quintuplet', note_duration)
    elif actual_notes == 6:
      note_duration = add_word('sextuplet', note_duration)
    
  process_note.last_note = note_duration
  
  return (note_pitch.strip(), note_duration.strip())
      

def generate_pitch_duration_pairs(xml_root):
  notes = xml_root.findall('.//note')
  divisions_per_quarter_note = int(xml_root.find('.//divisions').text)

  pairs = [process_note(divisions_per_quarter_note, note) for note in notes]
    
  if None in pairs:
    return None
    
  return pairs
  

def dump_pairs_to_file(pairs, filename):
  with open(filename, 'w') as f:
    #f.write(str(pairs))
    #f.write('\n')
    #return
    
    # this could be useful someday
    for pitch, duration in pairs:
      f.write(' '.join(pitch.split(' ')))
      f.write(' ')
      f.write(' '.join(duration.split(' ')))
      f.write(' ')
      #f.write('\n')  


def main(args):
  source_directory = args.src
  dest_directory = args.dst

  for root_dir, dirs, files in os.walk(source_directory):
    for f in tqdm(files):
      current_file = os.path.join(root_dir, f)  
      filename, file_extension = os.path.splitext(current_file)
      filename = os.path.basename(filename)
    
      # can only process MusicXML files
      if file_extension != '.xml':
        continue
      
      xml_root = ET.parse(current_file)
      
      pairs = None
      try:
        pairs = generate_pitch_duration_pairs(xml_root)
      except Exception as e:
        pass
      if pairs is None:
        print('Skipping: something is unsupported, perhaps double flats or unrecognized duration or there was some error' + filename)
      else:
        dump_pairs_to_file(pairs, os.path.join(dest_directory, filename + '.labels'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog=os.sys.argv[0], usage='python %(prog)s [options]')
  parser.add_argument('--src', '-s', help='Folder containing MusicXML (.xml) files to process')
  parser.add_argument('--dst', '-d', help='Folder where to put the output files containing pairs (pitch, duration)')

  main(parser.parse_args())    
  
