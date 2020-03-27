import json
import pandas as pd
from collections import defaultdict
import os

import config
import utils
from kgtk.cskg_utils import flatten_multiple_values, extract_label_aliases

VERSION=config.VERSION

NODE_COLS=config.nodes_cols
EDGE_COLS=config.edges_cols
datasource=config.wn_ds

# INPUT FILES
input_dir='../input/wordnet'
subclass_file='%s/output.csv' % input_dir

# OUTPUT FILES
output_dir='../output_v%s/wordnet' % VERSION
nodes_file='%s/nodes_v%s.csv' % (output_dir, VERSION)
edges_file='%s/edges_v%s.csv' % (output_dir, VERSION)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

### Store subclass edges ###

df=pd.read_csv(subclass_file, sep='\t', header=0, converters={5: json.loads})

tmp_edges_df=df[(df['object'] != 'None') & (df['subject'] != 'None')]
print('Initial number of edges:', len(tmp_edges_df))

clean_edges=flatten_multiple_values(tmp_edges_df, 'object')
print('Number of edges after flattening objects with multiple values:', len(clean_edges))

edges_df=pd.DataFrame(clean_edges, columns=EDGE_COLS)
edges_df['datasource']=datasource
edges_df.sort_values(by=['subject', 'predicate','object']).to_csv(edges_file, index=False, sep='\t')

### Create nodes file and store it ###
nodes=set()
for i, row in edges_df.iterrows():
    nodes.add(row['subject'])
    nodes.add(row['object'])
print(len(nodes), 'nodes in the edges file')

nodes_df=utils.create_df_with_wordnet_nodes(nodes, datasource, NODE_COLS)
nodes_df['datasource']=datasource
nodes_df.sort_values('id').to_csv(nodes_file, index=False, sep='\t')
