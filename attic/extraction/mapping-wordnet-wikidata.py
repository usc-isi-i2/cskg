from nltk.corpus import wordnet as wn
import pandas as pd
import json
import os

import config
import utils

data_source=config.mw_ds
weight=1.0
VERSION=config.VERSION

EDGE_COLS=config.edges_cols

# INPUT FILE
mapping_file='../input/mappings/Edges_WordNet2Wikidata_New.csv'

# OUTPUT FILE
output_dir=f'../output_v{VERSION}/mappings'
edges_file=f'{output_dir}/wn_wdt_mappings.csv'

MOWGLI_NS=config.mowgli_ns
WORDNET_NS=config.wordnet_ns

SAMEAS_REL=utils.create_uri(MOWGLI_NS, config.sameas)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

edges_df=pd.read_csv(mapping_file, sep='\t', header=0, converters={5: json.loads})
edges_df.sort_values(by=['subject', 'predicate','object']).to_csv(edges_file, index=False, sep='\t')

print(len(edges_df), 'edges stored')
