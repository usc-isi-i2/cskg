import pandas as pd
from nltk.corpus import wordnet as wn
import conceptnet_uri as cn
from collections import defaultdict
import json

from kgtk.cskg_utils import extract_label_aliases

def combine_dicts(dicts):
    new_dict=defaultdict(set)
    for d in dicts:
        for k in d.keys():
            new_dict[k] = new_dict[k] | set(d[k])
    for k, v in new_dict.items():
        v=list(v)
    return dict(new_dict)

def merge_and_deduplicate(x):
    return ','.join(list(set(x.split(','))))

def deduplicate_with_transformations(df, join_columns, transformations={'label': ','.join, 'aliases': ','.join, 'pos': ','.join, 'datasource': ','.join, 'other': ','.join}):
    grouped=df.groupby(join_columns, as_index=False).agg(transformations)
    for col in transformations.keys():
        if col not in ['other', 'weight']:
            grouped[col] = grouped[col].apply(merge_and_deduplicate)
        elif col=='other':
            grouped[col] = grouped[col].apply(combine_dicts)

    print(grouped)
    return grouped

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

