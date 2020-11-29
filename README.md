# CSKG
Scripts to create and utilize the commonsense knowledge graph: CSKG.

## Documentation

### Code structure

* `wikidata/` - scripts to extract the commonsense subset of Wikidata: `Wikidata-CS` 
* `visualgenome/` - scripts to investigate how to best extract and represent commonsense knowledge from Visual Genome
* `consolidation/` - scripts to consolidate CSKG
* `examples/` - notebooks that perform statistical analysis of CSKG, compute embeddings, etc.

### Data

The data is organized as follows:
* `input/` - individual data sources, not provided in the repo
* `output/` - output folder, created by the consolidation or embeddings scripts
* `tmp/` - temporary folder for keeping intermediate data


## Setup

```
conda create --name mowgli --file requirements.txt
conda activate mowgli
```

### Contact
Filip Ilievski (ilievski@isi.edu)
