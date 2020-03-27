#
## Makefile
# Masao Someki, 2019-01-30
#

.PHONY: all clean

all: venv gpu


venv:
	python3 -m venv venv
		. venv/bin/activate; pip install -U pip; pip install -r requirements.txt

gpu:
	. venv/bin/activate; pip install torch torchvision

clean:
	rm -fr venv dist build *.egg-info word2vec.*

#vim:ft=make
