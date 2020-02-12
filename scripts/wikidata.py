import json
import pandas as pd
from collections import defaultdict
import os
from nltk.corpus import wordnet as wn
from copy import copy

import config

VERSION=config.VERSION

NODE_COLS=config.nodes_cols
EDGE_COLS=config.edges_cols
datasource=config.wdt_ds

# INPUT FILES
input_dir='../input/wikidata'
subclass_file='%s/edges_wikidata_subclasses.csv' % input_dir
input_nodes_file='%s/wikidata-nodes.csv' % input_dir

# OUTPUT FILES
output_dir='../output_v%s/wikidata' % VERSION
nodes_file='%s/nodes_v%s.csv' % (output_dir, VERSION)
edges_file='%s/edges_v%s.csv' % (output_dir, VERSION)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
tmp_edges_df=pd.read_csv(subclass_file, sep='\t', header=0, converters={5: eval})

print(len(tmp_edges_df))

nodes=set()
for i, row in tmp_edges_df.iterrows():
    nodes.add(row['subject'])
    nodes.add(row['object'])
    
print(len(nodes))

rows=[]
for n in nodes:
    a_row=[n, "", "", "", datasource, {}]
    rows.append(a_row)

#tmp_nodes_df=pd.read_csv(input_nodes_file, sep='\t', header=0)#, converters={5: eval})

#nodes_df=tmp_nodes_df[tmp_nodes_df['id'].isin(nodes)]

nodes_df=pd.DataFrame(rows, columns = NODE_COLS)
nodes_df.sort_values('id').to_csv(nodes_file, index=False, sep='\t')

edges_df=tmp_edges_df
edges_df.sort_values(by=['subject', 'predicate','object']).to_csv(edges_file, index=False, sep='\t')