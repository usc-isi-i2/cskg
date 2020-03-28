import config
import json
import pandas as pd
import os
from kgtk.cskg_utils import collapse_identical_nodes, deduplicate_with_transformations 

def normalize_dicts(d):
	new_d={}
	for a_dict in d:		
		if a_dict:
			for k,v in a_dict.items():
				new_d[k]=v
	return new_d

def normalize_rows(nodes):
	new_rows=[]
	for i, row in nodes.iterrows():
		all_labels=row['label'].split(',') + row['aliases'].split(',')
		all_labels=list(set(all_labels))
		filtered=[s for s in all_labels if s]
		if len(filtered):
			main_label, *aliases=filtered
			row['label']=main_label
			row['aliases']=','.join(aliases)
		
		row['other']=normalize_dicts(row['other'])

		new_rows.append(row)
		if i%100000==0: print('processed', i)
	return pd.DataFrame(new_rows, columns=config.nodes_cols)

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
edge_transformations={'weight': max,  'datasource': ','.join,  'other': list}
collapsed_edges=deduplicate_with_transformations(collapsed_edges, ['subject', 'predicate', 'object'], edge_transformations)

#collapsed_nodes=normalize_rows(collapsed_nodes)
collapsed_nodes.sort_values('id').to_csv(merged_nodes_file, index=False, sep='\t')
collapsed_edges.sort_values(by=['subject', 'predicate','object']).to_csv(merged_edges_file, index=False, sep='\t')

