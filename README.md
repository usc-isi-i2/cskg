# cskg
Scripts to create the commonsense knowledge graph: CSKG

### Repo structure

* `analysis/` - scripts to analyze CSKG and the individual sources
* `extraction/` - scripts to extract and combine the individual sources. The bash script `run_all.sh` executes all scripts in a sequence.
* `input/` - individual data sources, not provided in the repo
* `output_v0xx` -> output folder, created by the extraction scripts

### Setup
(to reinstall, add `--upgrade --force-reinstall `)

* `pip install git+git://github.com/usc-isi-i2/kgtk`
* `conda install -c conda-forge graph-tool`
* `pip install -r requirements.txt`
* Install NLTK's packages 'framenet_v17' and 'wordnet', by typing this in your python3.7 console
```
import nltk
nltk.download('framenet_v17')
nltk.download('wordnet')
```

### Data
* Download ConceptNet (current version on Dec 18th 2019 is 5.7) from [here](https://github.com/commonsense/conceptnet5/wiki/Downloads) and unpack the assertions file ('conceptnet-assertions-5.7.0.csv') in `input/conceptnet` folder
* Download Visual Genome and store in `input/visualgenome`


### Contact
Filip Ilievski (ilievski@isi.edu)
