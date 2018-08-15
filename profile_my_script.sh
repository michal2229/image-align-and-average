#!/usr/bin/env bash

scriptname="py_image_merger.py"
profilefilename="$scriptname.prof"

python3 -m cProfile -o "$profilefilename" "$scriptname"

snakeviz "$profilefilename"