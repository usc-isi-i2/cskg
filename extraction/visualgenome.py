# VisualGenome
# version newest (v1.4, though the graphs are same as in v1.2)

### Setting imports, constants, and paths ###

import sys
sys.path.append('../')
import json
import pandas as pd
from collections import defaultdict
import os

import config
from utils import create_uri
from kgtk.cskg_utils import add_lowercase_labels, deduplicate_with_transformations

def add_relationships_data(rels, obj2names, image_id, all_nodes, all_edges, wn2label, wn2image):

    image_node=create_uri(VG_NS, 'I' + image_id)
#    image_metadata={'image_ids': [image_id]}
    image_metadata={}
    for rel in rels:
        synsets=rel['synsets']
        pred=rel['predicate']
        pred_id=create_uri(VG_NS, pred.replace(' ', '_'))
        sub_names=obj2names[str(rel['subject_id'])]
        obj_names=obj2names[str(rel['object_id'])]

        for sub_name in sub_names:
            sub_id=create_uri(VG_NS, sub_name.replace(' ', '_'))
            for obj_name in obj_names:
                obj_id=create_uri(VG_NS, obj_name.replace(' ', '_'))

                rel_edge=[pred_id, create_uri(VG_NS, 'subject'), sub_id, data_source, weight, image_metadata]
                all_edges.append(rel_edge)

                rel_edge=[pred_id, create_uri(VG_NS, 'object'), obj_id, data_source, weight, image_metadata]
                all_edges.append(rel_edge)

        pos=''
        for s in synsets:
            rel_syn_col=[pred_id, WORDNET_SENSE_REL, create_uri(WORDNET_NS, s), data_source, weight, image_metadata]
            all_edges.append(rel_syn_col)
            
            wn2label[s].add(pred)
            wn2image[s].add(image_id)

            pos=s.split('.')[-2]
            
            all_rel_synsets.append(s)
        
        # CREATE relationship node
        label, aliases=add_lowercase_labels([pred])
        rel_node=[pred_id, label, ','.join(aliases), pos, data_source, image_metadata]
        all_nodes.append(rel_node)

        rel_img_edge=[pred_id, INIMAGE_REL, image_node, data_source, weight, image_metadata]
        all_edges.append(rel_img_edge)

    return all_nodes, all_edges, wn2label, wn2image

def add_object_data(objects,
                  image_id,
                  all_nodes,
                  all_edges,
                  wn2label,
                  wn2image):
    obj2names={}
    image_node=create_uri(VG_NS, 'I' + image_id)
#    image_metadata={'image_ids': [image_id]}
    image_metadata={}
    for o in objects:
        for name in o['names']:
            o_id=create_uri(VG_NS, name.replace(' ', '_'))

            o_pos=''
            if 'attributes' in o.keys():
                for attr in o['attributes']:
                    a_id=create_uri(VG_NS, attr.replace(' ', '_'))
                    a_pos=''
                    if attr in attr_synsets:
                        a_synset=attr_synsets[attr]
                        attr_wn_edge=[a_id, WORDNET_SENSE_REL, create_uri(WORDNET_NS, a_synset),
                              data_source, weight, image_metadata]
                        all_edges.append(attr_wn_edge)

                        # save wordnet data for an attribute
                        wn2label[a_synset].add(attr)
                        wn2image[a_synset].add(image_id)
                        a_pos=a_synset.split('.')[-2]
                        o_pos=a_pos

                    # attribute node
                    a_label, a_aliases=add_lowercase_labels([attr])
                    attr_node=[a_id, a_label, ','.join(a_aliases), a_pos, data_source, image_metadata]
                    all_nodes.append(attr_node)

                    # edge from object to an attribute
                    obj_attr_edge=[o_id, HAS_PROPERTY_REL, a_id, data_source, weight, image_metadata]
                    all_edges.append(obj_attr_edge)


            obj_node=[o_id, name, '', o_pos, data_source, image_metadata] 
            all_nodes.append(obj_node)

        obj_img_edge=[o_id, INIMAGE_REL, image_node, data_source, weight, image_metadata]
        all_edges.append(obj_img_edge)

        obj2names[str(o['object_id'])]=o['names']

    return all_nodes, all_edges, wn2label, wn2image, obj2names

VERSION=config.VERSION

# INPUT FILES
input_dir='../input/visualgenome'
attr_synsets_path='%s/attribute_synsets.json' % input_dir
vg_scene_path='%s/scene_graphs.json' % input_dir
vg_regions_path='%s/region_graphs.json' % input_dir

# OUTPUT FILES
output_dir='../output_v%s/visualgenome' % VERSION
nodes_file='%s/nodes_v%s.csv' % (output_dir, VERSION)
edges_file='%s/edges_v%s.csv' % (output_dir, VERSION)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(vg_scene_path, 'r') as f:
    images_data=json.load(f)

with open(attr_synsets_path, 'r') as f:
    attr_synsets=json.load(f)

print('num images', len(images_data))

NODE_COLS=config.nodes_cols
EDGE_COLS=config.edges_cols

MOWGLI_NS=config.mowgli_ns
WORDNET_NS=config.wordnet_ns
VG_NS=config.visualgenome_ns

WORDNET_SENSE_REL=create_uri(VG_NS, config.pwordnet_sense)
SUBJECT_REL=create_uri(VG_NS, config.subject)
OBJECT_REL=create_uri(VG_NS, config.objct)
INIMAGE_REL=create_uri(VG_NS, config.in_image)

HAS_PROPERTY_REL=config.has_prop

CUSTOM_DATASET=config.custom_dataset

data_source=config.vg_ds
weight=1.0

debug=False

### Load the data into two tables: nodes (from objects with attributes) and edges (from relationships) WITH deduplication ###
#### Process edges first ####

all_edges=[]
all_nodes=[]

wn2label=defaultdict(set)
wn2image=defaultdict(set)

all_rel_synsets=[]
all_obj_synsets=[]
all_attr_synsets=[]
preds=[]

for counter, an_image in enumerate(images_data):    
    image_id=str(an_image['image_id'])
#    image_metadata={'image_ids': [image_id]}
    image_metadata={}

    # OBJECTS
    all_nodes, all_edges,  wn2label, wn2image, obj2labels = add_object_data(an_image['objects'],
                                                                                      image_id,
                                                                                      all_nodes,
                                                                                      all_edges,
                                                                                      wn2label,
                                                                                      wn2image)

    # RELATIONSHIPS
    all_nodes, all_edges, wn2label, wn2image = add_relationships_data(an_image['relationships'],
                                                                                      obj2labels,
                                                                                      image_id, 
                                                                                      all_nodes, 
                                                                                      all_edges,
                                                                                      wn2label,
                                                                                      wn2image)

    image_node=create_uri(VG_NS, 'I' + image_id)
    image_node_entry=[image_node, '', '', '', data_source, image_metadata] 
    all_nodes.append(image_node_entry)

    if debug:
        print(an_image)
        print()
        print(all_nodes)
        print()
        print(all_edges)
        break
    

print('number of synsets', len(set(all_rel_synsets) | set(all_attr_synsets) | set(all_obj_synsets)))

#### Add the synset data to the nodes.csv file ####

for synset, labels in wn2label.items():
    label, aliases=add_lowercase_labels(labels)
    pos=synset.split('.')[-2]
    images=list(wn2image[synset])
    image_metadata={}
    wn_node=[create_uri(WORDNET_NS, synset), label, ','.join(aliases), pos, data_source, image_metadata] 
    all_nodes.append(wn_node)

print('num nodes', len(all_nodes))
print('num edges', len(all_edges))
            
nodes_df=pd.DataFrame(all_nodes, columns = NODE_COLS)

# Drop duplicates
#nodes_df.drop_duplicates(subset='id', keep = 'first', inplace = True)

node_transformations={'label': ','.join, 'aliases': ','.join, 'pos': ','.join, 'datasource': ','.join, 'other': list}
nodes_df=deduplicate_with_transformations(nodes_df, 'id', node_transformations)

#nodes_df=nodes_df.groupby(['id'], as_index=False).agg({'aliases': ','.join})

print('combined nodes after deduplication:', len(nodes_df))

nodes_df.sort_values('id').to_csv(nodes_file, index=False, sep='\t')

edges_df = pd.DataFrame(all_edges, columns = EDGE_COLS)
# Drop duplicates
#edges_df.drop_duplicates(subset =["subject", "predicate", "object"],
#keep = 'first', inplace = True)

edge_transformations={'weight': max,  'datasource': ','.join,  'other': list}
edges_df=deduplicate_with_transformations(edges_df, ["subject", "predicate", "object"], edge_transformations)

print('combined edges after deduplication:', len(edges_df))
edges_df.sort_values(by=['subject', 'predicate','object']).to_csv(edges_file, index=False, sep='\t')
