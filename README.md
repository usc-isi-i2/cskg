# CSKG: The CommonSense Knowledge Graph

[![doi](https://zenodo.org/badge/DOI/10.5281/zenodo.4331372.svg)](https://doi.org/10.5281/zenodo.4331372) [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]


[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

CSKG is a commonsense knowledge graph that combines seven popular sources into a consolidated representation:
* [ATOMIC](https://homes.cs.washington.edu/~msap/atomic/)
* [ConceptNet](http://conceptnet.io/)
* [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/)
* [Roget](http://www.roget.org/)
* [Visual Genome](http://visualgenome.org/)
* [Wikidata](http://wikidata.org/) (We use the [Wikidata-CS](https://zenodo.org/record/3983030#.YEkr45NKimk) subset)
* [WordNet](https://wordnet.princeton.edu/)

CSKG is represented as a hyper-relational graph, by using the  KGTK [data model](https://kgtk.readthedocs.io/en/latest/data_model/) and [file specification](https://kgtk.readthedocs.io/en/latest/specification/). Its [creation](https://github.com/usc-isi-i2/cskg/blob/master/consolidation/create_cskg.sh) is entirely supported by KGTK operations.


CSKG is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

## Getting started

### Documentation
* [https://cskg.readthedocs.io/en/latest/](https://cskg.readthedocs.io/en/latest/)
* [Tutorial on Commonsense Knowledge Graphs (ISWC'20)](https://usc-isi-i2.github.io/ISWC20/)

### Data
* [CSKG](https://doi.org/10.5281/zenodo.4331372)
* [CSKG with dimension info](https://drive.google.com/file/d/1vj9Djf7V-lXunWDbsO7vwqS-YTGftPbq/view?usp=sharing)

### Embeddings
* [CSKG embeddings on google drive](https://drive.google.com/drive/u/1/folders/16347KHSloJJZIbgC9V5gH7_pRx0CzjPQ)

## Consolidating your own CSKG

1. Setup your conda environment
```
conda create --name mowgli --file requirements.txt
conda activate mowgli
```

2. Download and store individual sources, except WordNet and FrameNet. By default, these should be stored in the `input` directory.

3. Customize and run [create_cskg.sh](https://github.com/usc-isi-i2/cskg/blob/master/consolidation/create_cskg.sh). 

## How to cite
```
@article{ilievski2021cskg,
  title={CSKG: The CommonSense Knowledge Graph},
  author={Ilievski, Filip and Szekely, Pedro and Zhang, Bin},
  journal={Extended Semantic Web Conference (ESWC)},
  year={2021}
}
```
