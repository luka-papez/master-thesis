"""
  This script will take all MusicXML (.xml) files from directory given as first argument
  and partition them into multiple MusicXML files with each new created file containing a single
  stave of the original sheet music. The created files are placed in the directory specified by second argument.
  Additionaly, all text (title, credit, lyrics, chords etc.) are removed in the newly created files.

  Author: Luka Papez
  Date: Spring 2018
"""

# TODO: ignore scores with bass or baritone key

import os
import sys
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm


def remove_text(xml_root):
  # TODO: remove all other text
  nodes_to_remove = [
    'text', 'words', 'lyric', 'credit', 'creator', 'work', 'rights',
    'harmony', 'direction', 'direction-type', 'movement-title'
    ]

  for node_name in nodes_to_remove:
    while True:
      node        = xml_root.find('.//{}'.format(node_name))
      node_parent = xml_root.find('.//{}/..'.format(node_name))
      if node_parent is not None:
        node_parent.remove(node)
      else:
        break

  return xml_root


def measure_is_in_a_new_row(measure):
  # measure in a new row apparently begins with 'print' node
  return measure.findall('.//print')


def split_into_staves(xml_root):
  all_measures = xml_root.findall('.//measure')

  staves = []
  for measure in all_measures:
    if measure_is_in_a_new_row(measure):
      staves.append([])

    current_stave = staves[-1]
    current_stave.append(measure)

  return staves


def insert_music_xml_metadata(stave_filename):
  # TODO: this is horribly inefficient but python XML writer always starts writing at line 0
  lines = []
  with open(stave_filename, 'r') as f:
    lines = f.readlines()

  lines.insert(0, '<?xml version="1.0" encoding="UTF-8"?>\n')
  lines.insert(1, '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n')
  with open(stave_filename, 'w') as f:
    for line in lines:
      f.write(line)


def write_staves_as_files(dest_directory, score_key, staves, xml_root):
  for stave_index, stave in enumerate(staves):
    part = xml_root.find('.//part')
    attributes = xml_root.find('.//attributes')

    part.clear()
    part.attrib['id'] = 'P1'

    for measure_number, measure in enumerate(stave):
      measure.attrib['number'] = str(measure_number + 1)

      # this is needed to preserve bass clef and other possible modifiers
      if measure_number == 0 and attributes is not None:
        measure.insert(0, attributes)

      part.append(measure)

    stave_filename = os.path.join(dest_directory, score_key + '_row_{}.xml'.format(stave_index))

    xml_root.write(stave_filename)

    # insert_music_xml_metadata(stave_filename)


def main(args):
  source_directory = args.src
  dest_directory = args.dst

  for root_dir, dirs, files in os.walk(source_directory):
    for f in tqdm(files):
      current_file = os.path.join(root_dir, f)
      filename, file_extension = os.path.splitext(current_file)

      # can only process MusicXML files
      if file_extension != '.xml':
        continue


      try:
        xml_root = ET.parse(current_file)
      except BaseException as e:
          print(current_file)


      # TODO: taking only one part into consideration for now
      extra_parts = xml_root.findall('.//part')[1:]
      part_parent = xml_root.find('.//part/..')
      for part in extra_parts:
        part_parent.remove(part)

      # 1)
      xml_root = remove_text(xml_root)

      # 2)
      staves = split_into_staves(xml_root)

      # 3)
      write_staves_as_files(dest_directory, os.path.basename(filename), staves, xml_root)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog=os.sys.argv[0], usage='python %(prog)s [options]')
  parser.add_argument('--src', '-s', help='Folder containing MusicXML (.mxl) files to process')
  parser.add_argument('--dst', '-d', help='Folder where to put the partitioned staves as MusicXML (.xml) files')

  main(parser.parse_args())
