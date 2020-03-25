import utils
import config

import pandas as pd

SYN_REL='/r/Synonym'
ANT_REL='/r/Antonym'

ROGET_NS='rg'
ROGET_DS='ROGET'
datasource=ROGET_DS
weight=1.0

NODE_COLS=config.nodes_cols
EDGE_COLS=config.edges_cols
VERSION=config.VERSION

input_dir='../input/roget'
synonyms_file='%s/synonyms.txt' % input_dir
antonyms_file='%s/antonyms.txt' % input_dir

output_dir='../output_v%s/roget' % VERSION
nodes_file='%s/nodes_v%s.csv' % (output_dir, VERSION)
edges_file='%s/edges_v%s.csv' % (output_dir, VERSION)

def create_cskg_rows(lines, ncols, ecols, predicate):
	nr=[]
	er=[]
	nodes=set()
	for l in lines:
		n1, n2=l.split()
		n1=n1.strip()
		n2=n2.strip()
		n1_urn=utils.create_uri(ROGET_NS, n1[3:])
		n2_urn=utils.create_uri(ROGET_NS, n2[3:])
		edge=[n1_urn, predicate, n2_urn, datasource, weight, {}]
		er.append(edge)
		if n1 not in nodes:
			lang, *lbl=n1.split('_')
			n1_label=' '.join(lbl)
			n1_aliases=''
			if '-' in n1_label:
				n1_aliases=' '.join(n1_label.split('-'))
			node=[n1_urn, n1_label, n1_aliases, '', datasource, {}]
			nr.append(node)
			nodes.add(n1)
		if n2 not in nodes:
			lang, *lbl=n2.split('_')
			n2_label=' '.join(lbl)
			n2_aliases=''
			if '-' in n2_label:
				n2_aliases=' '.join(n2_label.split('-'))
			node=[n2_urn, n2_label, n2_aliases, '', datasource, {}]
			nr.append(node)
			nodes.add(n2)

	return nr, er

with open(synonyms_file, 'r') as f:
	syn_node_rows, syn_edge_rows=create_cskg_rows(f, NODE_COLS, EDGE_COLS, SYN_REL)

with open(antonyms_file, 'r') as f:
    ant_node_rows, ant_edge_rows=create_cskg_rows(f, NODE_COLS, EDGE_COLS, ANT_REL)


node_rows=syn_node_rows + ant_node_rows
edge_rows=syn_edge_rows + ant_edge_rows

nodes_df=pd.DataFrame(node_rows, columns=NODE_COLS)
edges_df=pd.DataFrame(edge_rows, columns=EDGE_COLS)

nodes_df.drop_duplicates(subset='id', keep = 'first', inplace = True)
edges_df.drop_duplicates(subset =["subject", "predicate", "object"],
	keep = 'first', inplace = True)

print(len(nodes_df), 'nodes')
print(len(edges_df), 'edges')

# store nodes and edges
nodes_df.sort_values('id').to_csv(nodes_file, index=False, sep='\t')
edges_df.sort_values(by=['subject', 'predicate','object']).to_csv(edges_file, index=False, sep='\t')


# generate mappings to conceptnet
