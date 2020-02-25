import sys
sys.path.append('../')

from nltk.corpus import wordnet as wn
import pandas as pd
import json
import os

import config
from utils import create_uri, extract_wn_version_id, map_v31_to_v30, create_df_with_wordnet_nodes

cnfile='../input/conceptnet/conceptnet-en-with-externalurl.csv'
data_source=config.wn_ds
weight="1.0"
VERSION=config.VERSION

EDGE_COLS=config.edges_cols
NODE_COLS=config.nodes_cols

cn_nodes_file=f'../output_v{VERSION}/conceptnet/nodes_v{VERSION}.csv'
wn_nodes_file=f'../output_v{VERSION}/wordnet/nodes_v{VERSION}.csv'
vg_nodes_file=f'../output_v{VERSION}/visualgenome/nodes_v{VERSION}.csv'

wordnet30_ili_file='../input/mappings/ili-map-pwn30.tab'
wordnet31_ili_file='../input/mappings/ili-map-pwn31.tab'

# OUTPUT FILE
output_dir=f'../output_v{VERSION}/mappings'
edges_file=f'{output_dir}/edges_v{VERSION}.csv'

MOWGLI_NS=config.mowgli_ns
WORDNET_NS=config.wordnet_ns

SAMEAS_REL=create_uri(MOWGLI_NS, config.sameas)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

### Load the CN filtered data in pandas ###

df=pd.read_csv(cnfile, sep='\t', header=None, converters={4: json.loads})
df.columns=['assertion','rel','subj','obj','metadata']
df.drop(columns=['assertion'])
print('size of df with external URLs', len(df))

df_wordnet=df.loc[(df['rel'] == '/r/ExternalURL') & (df['obj'].str.contains(r'http://wordnet-'))]
print('size of df with wordnet external links', len(df_wordnet))

### Get WordNet nodes ###

all_nodes=set()
wn_nodes=set()

with open(cn_nodes_file, 'r') as f:
    for line in f:
        first=line.split('\t')[0]
        all_nodes.add(first)

with open(wn_nodes_file, 'r') as f:
    for line in f:
        first=line.split('\t')[0]
        all_nodes.add(first)
        wn_nodes.add(first)
wn_nodes_df=pd.read_csv(wn_nodes_file, sep='\t', header=0, converters={5: eval})

with open(vg_nodes_file, 'r') as f:
    for line in f:
        first=line.split('\t')[0]
        if first.startswith('wn:'):
            all_nodes.add(first)
            wn_nodes.add(first)

print('wordnet30 nodes:', len(wn_nodes))

### Load Wordnet mapings 3.0 to 3.1 ###

mapping_31_30=map_v31_to_v30(wordnet31_ili_file, wordnet30_ili_file)
print('v31 to v30 mappings', len(mapping_31_30))

not_in_ili_nodes=set()
missing_nodes=set()
all_edges=[]
for i, row in df_wordnet.iterrows():
    wn_version, wn_offset_id=extract_wn_version_id(row['obj'])
    if wn_version=='wn31':
        if wn_offset_id not in mapping_31_30.keys():
            not_in_ili_nodes.add(wn_offset_id)
        else:
            wn_30_id=mapping_31_30[wn_offset_id]
            offset, pos=wn_30_id.split('-')
            wn_30_synset=wn.synset_from_pos_and_offset(pos,int(offset)).name()
            wn_30_synset_uri=create_uri(WORDNET_NS, wn_30_synset)

            #if row['subj'] in all_nodes:# and
            if wn_30_synset_uri in wn_nodes:
                an_edge=[row['subj'], SAMEAS_REL, wn_30_synset_uri, data_source, weight, {}]
                all_edges.append(an_edge)
            else:
                missing_nodes.add(wn_30_synset_uri)
    else:
        print('Other WN version', wn_version)

print('missing ILI index nodes version 31', len(not_in_ili_nodes))
print('nodes missing in the node lists of WordNet and VG', len(missing_nodes))

missing_nodes_df=create_df_with_wordnet_nodes(missing_nodes, data_source, NODE_COLS)
combined_wn_nodes_df=pd.concat([wn_nodes_df, missing_nodes_df])
combined_wn_nodes_df.sort_values('id').to_csv(wn_nodes_file, index=False, sep='\t')

edges_df = pd.DataFrame(all_edges, columns = EDGE_COLS)
edges_df.sort_values(by=['subject', 'predicate','object']).to_csv(edges_file, index=False, sep='\t')

print('Number of edges', len(edges_df))
