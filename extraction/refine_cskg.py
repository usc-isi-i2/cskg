import config
import json
import pandas as pd
import os
from kgtk.cskg_utils import collapse_identical_nodes 

def normalize_labels(nodes):
	new_rows=[]
	for i, row in nodes.iterrows():
		if ',' in row['label']:
			main_label, *aliases=row['label'].split(',')
			if main_label=='':
				print(aliases)
			row['label']=main_label
			prev_aliases=row['aliases'].split(',')
			all_aliases=set(prev_aliases) | set(aliases)
			row['aliases']=','.join(list(all_aliases))
		new_rows.append(row)
	return pd.DataFrame(new_rows, columns=config.node_columns)

VERSION=config.VERSION
cskg_dir='../output_v%s/cskg' % VERSION
output_merged_dir='%s_merged' % cskg_dir

cskg_nodes_file='%s/nodes_v%s.csv' % (cskg_dir, VERSION)
merged_nodes_file='%s/nodes_v%s.csv' % (output_merged_dir, VERSION)

cskg_edges_file='%s/edges_v%s.csv' % (cskg_dir, VERSION)
merged_edges_file='%s/edges_v%s.csv' % (output_merged_dir, VERSION)

if not os.path.exists(output_merged_dir):
    os.makedirs(output_merged_dir)

### Collapse same-as relations/nodes ###

collapsed_edges, collapsed_nodes = collapse_identical_nodes(cskg_edges_file, cskg_nodes_file)

collapsed_nodes=normalize_labels(collapsed_nodes)
collapsed_nodes.sort_values('id').to_csv(merged_nodes_file, index=False, sep='\t')
collapsed_edges.sort_values(by=['subject', 'predicate','object']).to_csv(merged_edges_file, index=False, sep='\t')

