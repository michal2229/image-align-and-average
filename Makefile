interpreter=python2
mainsrc=py_image_merger.py

all: main

main: *.py
	$(interpreter) $(mainsrc)

