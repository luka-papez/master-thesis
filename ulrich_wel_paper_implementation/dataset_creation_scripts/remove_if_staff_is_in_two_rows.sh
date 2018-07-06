#!/bin/bash

function remove_if_staff_is_in_two_rows {
  RATIO=4

  #IMAGE_SIZE="$(identify $1 | cut -d' ' -f3)"
  IMAGE_SIZE="$(./magick/identify.exe $1 | cut -d' ' -f3)"
  IMAGE_WIDTH=" $(echo $IMAGE_SIZE | cut -d'x' -f1)"
  IMAGE_HEIGTH="$(echo $IMAGE_SIZE | cut -d'x' -f2)"

  VALID=$(echo "$IMAGE_HEIGTH * $RATIO < $IMAGE_WIDTH" | bc -l)

  if [ "$VALID" = "0" ]; then
    echo "Removing $1 because of wrong aspect ratio - score has two rows"
    rm $1
  fi
}

remove_if_staff_is_in_two_rows $1

# TODO: something more elaborate
