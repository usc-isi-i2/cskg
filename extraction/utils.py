import pandas as pd
from nltk.corpus import wordnet as wn
import conceptnet_uri as cn
import json

import config
from kgtk.cskg_utils import extract_label_aliases

weight=1.0
mowgli_ds=config.mw_ds
EDGE_COLS=config.edges_cols

def sameas_to_conceptnet(other_nodes_file, cn_nodes_file, other_prefix, edges_file):
    cn_nodes_df=pd.read_csv(cn_nodes_file, sep='\t', header=0)
    other_nodes_df=pd.read_csv(other_nodes_file, sep='\t', header=0)#, converters={5: json.loads})


    cn_nodes=set()
    for i, n in cn_nodes_df.iterrows():
        cn_nodes.add(n['id'].replace('/c/en/', ''))
             
    other_nodes=set()
    for i, v in other_nodes_df.iterrows():
        other_nodes.add(v['id'].replace(other_prefix, ''))

    common_nodes=cn_nodes & other_nodes

    mapping_rows=[]
    for common in common_nodes:
        other_node='%s:%s' % (other_prefix, common)
        cn_node='/c/en/%s' % common
        a_row=[other_node, 'mw:SameAs', cn_node, mowgli_ds, weight, {}]
        mapping_rows.append(a_row)

    print(len(mapping_rows))

    edges_df=pd.DataFrame(mapping_rows, columns=EDGE_COLS)
    edges_df.sort_values(by=['subject', 'predicate','object']).to_csv(edges_file, index=False, sep='\t')

def create_uri(ns, rel):
    return '%s:%s' % (ns, rel)

def get_cn_pos_tag(uri, MOWGLI_NS, POS_MAPPING):
    components=cn.split_uri(uri)
    if len(components)<4:
        return '', ''
    else:
        raw_pos=components[3]
        mapped_pos=create_uri(MOWGLI_NS, POS_MAPPING[raw_pos])
        return mapped_pos, raw_pos

def obtain_wordnet_lemmas(n):
    lemmas=[]
    syn=wn.synset(n)
    for lemma in syn.lemmas():
        lemmas.append(str(lemma.name()))
    return lemmas

def create_df_with_wordnet_nodes(nodes, datasource, node_columns):
    node_data=[]
    for a_node in nodes:
        n=a_node.split(':')[1]
        lemmas=obtain_wordnet_lemmas(n)

        label, aliases=extract_label_aliases(lemmas)
     
        if len(n.split('.'))>=3:
            pos=n.split('.')[-2]
        else:
            print('Warning: Too little values in a synset:', n)

        other={}
        a_row=[a_node, label, aliases, pos, datasource, other]
        node_data.append(a_row)

    print(len(node_data), 'nodes stored')

    nodes_df=pd.DataFrame(node_data, columns = node_columns)
    return nodes_df

def extract_wn_version_id(uri):
    splitted=uri.split('/')
    wn_offset_id=splitted[4][1:]
    return splitted[3], wn_offset_id

def map_v31_to_v30(wordnet31_ili_file, wordnet30_ili_file):
    mapping={}

    with open(wordnet31_ili_file, 'r') as f:
        for line in f:
            ili, wn31=line.split('\t')
            mapping[ili]={'31': wn31}

    with open(wordnet30_ili_file, 'r') as f:
        for line in f:
            ili, wn30=line.split('\t')
            if ili in mapping.keys():
                mapping[ili]['30']=wn30

    mapping_31_30={}
    for ili, ili_data in mapping.items():
        id_31=ili_data['31'].strip()
        id_30=ili_data['30'].strip()
        mapping_31_30[id_31]=id_30

    return mapping_31_30

