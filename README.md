# CSKG: The CommonSense Knowledge Graph

[![doi](https://zenodo.org/badge/DOI/10.5281/zenodo.4331372.svg)](https://doi.org/10.5281/zenodo.4331372)

CSKG is a commonsense knowledge graph that combines seven popular sources into a consolidated representation:
* ATOMIC
* ConceptNet
* FrameNet
* Roget
* Visual Genome
* Wikidata-CS
* WordNet

CSKG is represented as a hyper-relational graph, by using the  KGTK [data model](https://kgtk.readthedocs.io/en/latest/data_model/) and [file specification](https://kgtk.readthedocs.io/en/latest/specification/). Its [creation](https://github.com/usc-isi-i2/cskg/blob/master/consolidation/create_cskg.sh) is entirely supported by KGTK operations.

## Getting started

### Documentation
* [https://cskg.readthedocs.io/en/latest/](https://cskg.readthedocs.io/en/latest/)


### Data
* [CSKG on Zenodo](https://doi.org/10.5281/zenodo.4331372)

### Embeddings
* [CSKG embeddings on google drive](https://drive.google.com/drive/u/1/folders/16347KHSloJJZIbgC9V5gH7_pRx0CzjPQ)

## License

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg



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
