# mowgli-uci
Code related to UCI MOWGLI project. Tested on Ubuntu 18.04.2 LTS.

## Getting Started
Our installations will be in a conda environment. If you don't have a conda installed, follow \[[link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)\] to install it.

Then set up your own conda environment and install packages.
```{bash}
conda create -n mowgli-env python=3.6 anaconda
source activate mowgli-env

pip install -r requirements.txt
conda install --yes faiss-cpu -c pytorch -n mowgli-env
python -m spacy download en_core_web_lg
```

You will also need to download the latest ConceptNet Numberbatch embeddings \[[link](https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz)\] and unzip it.

## Graphify Instructions
To convert a text file of sentences (each line has a string) to a JSONL file of graphs (each line has a JSON graph), run the following:
```{Python}
python graphify.py
	--input [INPUT_FILE] # A text file with a sentence (or sentences) per line
	--output [OUTPUT_FILE] # A JSONL file with one graph per line
```

Documentation for how the graph is created is available in `graphify_documentation/`.

## Linking Instructions

To obtain link candidates run the following (the arguments need to be in quotes).
The input file should be the output of the `graphify.py` file.
```{Python}
python link.py link \
	--input [INPUT FILE] \
	--output [OUTPUT FILE] \
	--embedding_file [PATH TO THE NUMBERBATCH EMBEDDINGS]
```

For further details and additional arguments see:
```{Python}
python link.py --help
```

## Demo
We have some files in `demo/` to illustrate what the outputs are these scripts look like.
The files in `demo/` were produced using the following commands:
```
python graphify.py --input demo/sentences.txt --output demo/graphs.jsonl
python link.py link --input demo/graphs.jsonl --output demo/linked_graphs.jsonl --embedding_file numberbatch-en-19.08.txt
```
