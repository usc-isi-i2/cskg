# CSKG
Scripts to create and utilize the commonsense knowledge graph: CSKG.

### Code structure

* `analysis/` - scripts to analyze CSKG and the individual sources
* `wikidata/` - scripts to extract a commonsense subset of Wikidata: `Wikidata-CS`
* `embeddings/` - scripts to create various CSKG embeddings


### Data

The data is organized as follows:
* `input/` - individual data sources, not provided in the repo
* `output_v0xx` -> output folder, created by the extraction scripts


### Embedding Usage
```
1. cd ../cskg
2. python embeddings/embedding_training.py --help   
3. python embeddings/embedding_training.py -i input/cskg_connected.tsv -o output/cskg_connected 
```

### Contact
Filip Ilievski (ilievski@isi.edu)
