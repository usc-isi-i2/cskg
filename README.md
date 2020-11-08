# CSKG
Scripts to create and utilize the commonsense knowledge graph: CSKG.

## Documentation

### Code structure

* `wikidata/` - scripts to extract the commonsense subset of Wikidata: `Wikidata-CS` 
* `visualgenome/` - scripts to investigate how to best extract and represent commonsense knowledge from Visual Genome
* `consolidation/` - scripts to consolidate CSKG
* `embeddings/` - scripts to create various CSKG embeddings
* `examples/` - notebooks that perform statistical analysis of CSKG, compute embeddings, etc.
* `test_code/` - additional code to test code

### Data

The data is organized as follows:
* `input/` - individual data sources, not provided in the repo
* `output_v0xx` -> output folder, created by the extraction scripts

### Embedding Usage
```
1. python embeddings/embedding_training.py --help   
2. python embeddings/embedding_training.py -i input/cskg_connected.tsv -o output/cskg_connected 
```

## Setup

```
conda create --name mowgli
conda activate mowgli
pip install -r requirements.txt
```

### Contact
Filip Ilievski (ilievski@isi.edu)
