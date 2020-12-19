# CSKG
Scripts to create and utilize the commonsense knowledge graph: CSKG.

## Resources
* [CSKG on Zenodo](https://doi.org/10.5281/zenodo.4331372)
* [CSKG embeddings on google drive](https://drive.google.com/drive/u/1/folders/16347KHSloJJZIbgC9V5gH7_pRx0CzjPQ)
* [Documentation](https://docs.google.com/document/d/1fbbqgyX0N2EdxLam6hatfke1R-nZWkoN6M1oB_f4aQo/edit?usp=sharing)

## License

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg


## Documentation

### Data

The data is organized as follows:
* `input/` - individual data sources, not provided in the repo
* `output/` - output folder, created by the consolidation or embeddings scripts
* `tmp/` - temporary folder for keeping intermediate data


## Setup
1. Setup your conda environment
```
conda create --name mowgli --file requirements.txt
conda activate mowgli
```

2. Download CSKG and its embeddings (see above)

### Contact
Filip Ilievski (ilievski@isi.edu)
