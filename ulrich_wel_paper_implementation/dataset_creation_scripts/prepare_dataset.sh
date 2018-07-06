#!/bin/bash

# This script does my master thesis. No joke.
#
# Author: Luka Papez (luka.papez@outlook.com)
# Date:   Spring 2018
#
# this script must be run in bash as it requires many bash-specific features
#
# First  argument: folder where to find input - decompressed MusicXML files
# Second argument: folder where to put individual rows (XML) of scores
# Third  argument: folder where to put dataset input images generated from row XMLs
# Fourth argument: folder where to put generated images with added noise
# Fifth  argument: folder where to put the dataset output labels generated from row XMLs

export QT_QPA_PLATFORM="offscreen"

# create output folders if they don't exist
mkdir -p $2 &> /dev/null
mkdir -p $3 &> /dev/null
mkdir -p $4 &> /dev/null
mkdir -p $5 &> /dev/null

# REQUIREMENTS: python 2.7+
#echo "PARTITIONING MUSICXML FILES TO SEPARATE STAVES"
#python partition_xml_scores_into_xml_rows.py --src $1 --dst $2

# REQUIREMENTS: MuseScore
#echo "CONVERTING PARTITIONED STAFF XMLs TO IMAGES"
#find $2 -type f -print -exec sh -c "mscore -s -m {} -o $3/"'$(basename {} .xml).png' 2> /dev/null \;

# REQUIREMENTS: imagemagick's convert
#echo "TRIMMING STAFF IMAGES BUT LEAVING A SMALL BORDER"
BORDER_SIZE="1%x6%"
#find $3 -name "*.png" -print -type f -exec ./magick/convert.exe {} -trim -bordercolor none -border $BORDER_SIZE {} \;
#find $3 -name "*.png" -print -type f -exec convert {} -trim -bordercolor none -border $BORDER_SIZE {} \;

# REQUIREMENTS: imagemagick's identify
#echo "REMOVING STAVES WHICH ARE ACROSS TWO ROWS DUE TO MUSESCORE BUG"
#find $3 -name "*.png" -type f -exec bash remove_if_staff_is_in_two_rows.sh {} \;

# REQUIREMENTS: imagemagick's mogrify
# the exclamation mark at the end forces the resize without keeping aspect ratio
IMAGE_HEIGHT=128
IMAGE_WIDTH=1024
#echo "DOWN-SIZING IMAGES TO HEIGHT: $IMAGE_HEIGHT, WIDTH: $IMAGE_WIDTH"
#find $3 -name "*.png" -type f -exec ./magick/convert.exe -resize "$IMAGE_WIDTH""x""$IMAGE_HEIGHT""^!" {} {} \;
#find $3 -name "*.png" -type f -exec convert -resize "$IMAGE_WIDTH""x""$IMAGE_HEIGHT""!" {} {} \;

# REQUIREMENTS: python 2.7+, numpy and OpenCV
#echo "ADDING NOISE TO STAFF IMAGES"
#python add_noise_to_images.py -n 1 --src $3 --dst $4

# REQUIREMENTS: python 2.7+
# TODO: remove irrelevant rows for which images have been removed
#echo "CONVERTING XML ROWS TO PITCH DURATION PAIRS"
#python convert_music_xml_to_pitch_duration_pairs.py --src $2 --dst $5
