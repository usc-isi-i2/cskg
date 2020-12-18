# CSKG
Scripts to create and utilize the commonsense knowledge graph: CSKG.

## License

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg


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
