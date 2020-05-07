import config
import json
import pandas as pd
import os
from kgtk.cskg_utils import collapse_identical_nodes, deduplicate_with_transformations 

VERSION=config.VERSION
cskg_dir='../output_v%s/cskg-raw' % VERSION
output_merged_dir='../output_v%s/cskg-refined' % VERSION

cskg_nodes_file='%s/nodes_v%s.csv' % (cskg_dir, VERSION)
merged_nodes_file='%s/nodes_v%s.csv' % (output_merged_dir, VERSION)

cskg_edges_file='%s/edges_v%s.csv' % (cskg_dir, VERSION)
merged_edges_file='%s/edges_v%s.csv' % (output_merged_dir, VERSION)

if not os.path.exists(output_merged_dir):
    os.makedirs(output_merged_dir)

### Collapse same-as relations/nodes ###

collapsed_edges, collapsed_nodes = collapse_identical_nodes(cskg_edges_file, cskg_nodes_file)
edge_transformations={'weight': max,  'datasource': ','.join,  'other': list}
collapsed_edges=deduplicate_with_transformations(collapsed_edges, ['subject', 'predicate', 'object'], edge_transformations)

collapsed_nodes.sort_values('id').to_csv(merged_nodes_file, index=False, sep='\t')
collapsed_edges.sort_values(by=['subject', 'predicate','object']).to_csv(merged_edges_file, columns=config.edges_cols, index=False, sep='\t')

