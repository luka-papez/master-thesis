#!/bin/bash

# This script will take compressed MusicXML files from the dataset directory
# and decompress them into actual XML files that can be manipulated.
#
# Author: Luka Papež
# Date: Spring 2018

# https://musescore.org/en/node/8970
# Another simpler way is to just set “QT_QPA_PLATFORM” environment variable to “offscreen” — no need for xvfb at all. Tested with MuseScore 2.1.
# copy fonts to wherever musescore says

mkdir -p ../ulrich_wel_dataset_decompressed/train
mkdir -p ../ulrich_wel_dataset_decompressed/test
mkdir -p ../ulrich_wel_dataset_decompressed/validation

for file in $(find "../ulrich_wel_dataset" -name '*.mxl'); do
  target="$(echo "$file" | cut -f3- -d'/' | cut -f1 -d'.')"
  targetfile="../ulrich_wel_dataset_decompressed/$target.xml"

  if [ ! -f "$targetfile" ]; then
    echo $targetfile
    /c/Program\ Files\ \(x86\)/MuseScore\ 2/bin/MuseScore.exe $file -o $targetfile
  # else
  #   echo "Skipping $target"
  fi

done
