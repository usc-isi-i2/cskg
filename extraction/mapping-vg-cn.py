import os
import json
import pandas as pd
import config

mowgli_ds=config.mw_ds

weight=1.0
VERSION=config.VERSION

EDGE_COLS=config.edges_cols

# INPUT FILE
cn_nodes_file='../output_v%s/conceptnet/nodes_v%s.csv' % (VERSION, VERSION)
vg_nodes_file='../output_v%s/visualgenome/nodes_v%s.csv' % (VERSION, VERSION)

# OUTPUT FILE
edges_file='../output_v%s/mappings/wn_wdt_mappings.csv' % VERSION

cn_nodes_df=pd.read_csv(cn_nodes_file, sep='\t', header=0)
vg_nodes_df=pd.read_csv(vg_nodes_file, sep='\t', header=0)#, converters={5: json.loads})


cn_nodes=set()
for i, n in cn_nodes_df.iterrows():
    cn_nodes.add(n['id'].replace('/c/en/', ''))
            
vg_nodes=set()
for i, v in vg_nodes_df.iterrows():
    vg_nodes.add(v['id'].replace('vg:', ''))

common_nodes=cn_nodes & vg_nodes

mapping_rows=[]
for common in common_nodes:
    vg_node='vg:%s' % common
    cn_node='/c/en/%s' % common
    a_row=[vg_node, 'mw:SameAs', cn_node, mowgli_ds, 1.0, {}]
    mapping_rows.append(a_row)

print(len(mapping_rows))

edges_df=pd.DataFrame(mapping_rows, columns=EDGE_COLS)
edges_df.sort_values(by=['subject', 'predicate','object']).to_csv(edges_file, index=False, sep='\t')

