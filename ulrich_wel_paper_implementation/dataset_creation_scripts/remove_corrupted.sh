#!/bin/bash

chmod -R +w ../ulrich_wel_dataset

mv "../ulrich_wel_dataset/$1.mxl" ../ulrich_wel_dataset_missing/corrupted

chmod -R -w ../ulrich_wel_dataset
