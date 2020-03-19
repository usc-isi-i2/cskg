import sys
sys.path.append('../')

import pandas as pd
import json
from collections import defaultdict
import copy
import os

from utils import create_uri, get_cn_pos_tag
import conceptnet_uri as cn

import config

VERSION=config.VERSION

NODE_COLS=config.nodes_cols
EDGE_COLS=config.edges_cols

MOWGLI_NS=config.mowgli_ns

POS_MAPPING=config.pos_mapping

POS_REL=config.has_pos
POS_FORM_REL=config.has_pos_form
IS_POS_FORM_OF_REL=config.is_pos_form_of
WORDNET_SENSE_REL=config.wordnet_sense

CUSTOM_DATASET=config.custom_dataset

data_source=config.cn_ds

cn_path='../input/conceptnet/conceptnet-en.csv'
# OUTPUT FILES
output_dir='../output_v%s/conceptnet' % VERSION
nodes_file='%s/nodes_v%s.csv' % (output_dir, VERSION)
edges_full_file='%s/edges_v%s.csv' % (output_dir, VERSION)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

custom_weight=1.0

### Load the data in pandas ###
df=pd.read_csv(cn_path, sep='\t', header=None, converters={4: json.loads})
df.columns=['assertion','rel','subj','obj','metadata']
df.drop(columns=['assertion'])

### Create nodes.csv and edges.csv ###
#### Let's first extract the main data into temporary structures. ####

node_datasets=defaultdict(set)
all_edges=[]

for i, row in df.iterrows():

    subj=row['subj']
    obj=row['obj']
    rel=row['rel']
    dataset=row['metadata']['dataset']
    weight=row['metadata']['weight']
    sentence=''

    node_datasets[subj].add(dataset)
    node_datasets[obj].add(dataset)

    other={'dataset': dataset}
    edge_data=[subj, rel, obj, data_source, weight, other]
    all_edges.append(edge_data)

#### a. Prepare and store nodes ####

all_nodes=[]
for n, datasets in node_datasets.items():
    label=cn.uri_to_label(n)
    aliases_list=[]
    aliases=','.join(aliases_list)
    mapped_pos, raw_pos=get_cn_pos_tag(n, MOWGLI_NS, POS_MAPPING)
    other={'datasets': list(datasets)}
    col=[n, label, aliases, raw_pos, data_source, other]
    all_nodes.append(col)

for raw_pos, mapped_pos in POS_MAPPING.items():
    mowgli_pos=create_uri(MOWGLI_NS, mapped_pos)
    col=[mowgli_pos, raw_pos, mapped_pos, '', '', {"datasets": [CUSTOM_DATASET]}]
    all_nodes.append(col)


nodes_df = pd.DataFrame(all_nodes, columns = NODE_COLS)
nodes_df.sort_values('id').to_csv(nodes_file, index=False, sep='\t')
print('unique POS tags', nodes_df['pos'].unique())
print(len(nodes_df), 'nodes')

#### b. Enrich and store edges ####

all_edges_enriched=copy.deepcopy(all_edges)
other={'dataset': CUSTOM_DATASET}
for i, row in nodes_df.iterrows():
    
    node_id=row['id']
    components=cn.split_uri(node_id)
    
    if len(components)==4:
        # add POS relations
        mapped_pos, raw_pos = get_cn_pos_tag(node_id, MOWGLI_NS, POS_MAPPING)
        edge=[node_id, create_uri(MOWGLI_NS, POS_REL), mapped_pos, data_source, custom_weight, other]
        all_edges_enriched.append(edge)
        
        le_node='/%s' % '/'.join(components[:3])
        if le_node in node_datasets.keys():
            # add pos-form relations (both-ways)
            edge=[le_node, create_uri(MOWGLI_NS, POS_FORM_REL), node_id, data_source, custom_weight, other]
            all_edges_enriched.append(edge)

            edge=[node_id, create_uri(MOWGLI_NS, IS_POS_FORM_OF_REL), le_node, data_source, custom_weight, other]
            all_edges_enriched.append(edge)
        
    elif len(components)>=5 and components[4]=='wn':
        # add OMW relations
        pos_node='/%s' % '/'.join(components[:4])
        if pos_node in node_datasets.keys():
            edge=[pos_node, create_uri(MOWGLI_NS, WORDNET_SENSE_REL), node_id, data_source, custom_weight, other]
            all_edges_enriched.append(edge)
    
edges_enriched_df = pd.DataFrame(all_edges_enriched, columns = EDGE_COLS)

#### c. Complement missing symmetric data ####
all_difs=[edges_enriched_df]
for sym_rel in config.symmetric_rels:
    #if sym_rel!='/r/LocatedNear': continue
        
    sub_df=edges_enriched_df[edges_enriched_df.predicate==sym_rel]
    sub_df['other']=""
    print(sym_rel, len(sub_df))
    
    so_df=sub_df[EDGE_COLS]
    
    os_df=sub_df[['object', 'predicate', 'subject', 'datasource', 'weight', 'other']]
    os_df.columns=EDGE_COLS
    
    the_diff=os_df.merge(so_df,indicator = True, 
                         how='left').loc[lambda x : x['_merge']!='both']
    
    the_diff['other']=json.dumps({'dataset': CUSTOM_DATASET})
    the_diff['other']=the_diff['other'].apply(json.loads)
    
    print(the_diff.columns)
    
    print(len(the_diff))
    print()
    all_difs.append(the_diff)

all_data=pd.concat(all_difs)
all_data=all_data[EDGE_COLS]
all_data.sort_values(by=['subject', 'predicate','object']).to_csv(edges_full_file,
                                                                  index=False,
                                                                  sep='\t')

print('Number of edges in the full conceptnet version', len(all_data))
