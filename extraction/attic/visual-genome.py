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
from kgtk.utils.cskg_utils import add_lowercase_labels


def add_relationships_data(rels, image_id, all_nodes, all_edges, wn2label, wn2image):
    for rel in rels:
        rel_id=create_uri(VG_NS, 'R' + str(rel['relationship_id']))
        sub_id=create_uri(VG_NS, 'O' + str(rel['subject_id']))
        obj_id=create_uri(VG_NS, 'O' + str(rel['object_id']))
        synsets=rel['synsets']
        pred=rel['predicate']
        
        # CREATE REL-SUBJECT and REL-OBJECT EDGES
        rel_subj_col=[rel_id, SUBJECT_REL, sub_id, data_source, weight, {'image_id': image_id}]
        all_edges.append(rel_subj_col)
        
        rel_obj_col=[rel_id, OBJECT_REL, obj_id, data_source, weight, {'image_id': image_id}]
        all_edges.append(rel_obj_col)
        
        pos=''
        for s in synsets:
            rel_syn_col=[rel_id, WORDNET_SENSE_REL, create_uri(WORDNET_NS, s), data_source, weight, {'image_id': image_id}]
            all_edges.append(rel_syn_col)
            
            wn2label[s].add(pred)
            wn2image[s].add(image_id)            

            pos=s.split('.')[-2]
            
            all_rel_synsets.append(s)
        
        # CREATE relationship node
        label, aliases=add_lowercase_labels([pred])
        rel_node=[rel_id, label, ','.join(aliases), pos, data_source, {'image_id': image_id}]
        all_nodes.append(rel_node)
           
    return all_nodes, all_edges, wn2label, wn2image

def add_attr_data(attrs, image_id, all_nodes, all_edges, wn2label, wn2image, attr_id):
    for a in attrs:
        a_id=create_uri(VG_NS, 'A' + str(attr_id))
        attr_id+=1

        # attribute-related edges
        obj_attr_edge=[obj_id, HAS_PROPERTY_REL, a_id, data_source, weight, {'image_id': image_id}]
        all_edges.append(obj_attr_edge)

        a_pos=''
        if a in attr_synsets:
            a_synset=attr_synsets[a]

            attr_wn_edge=[a_id, WORDNET_SENSE_REL, create_uri(WORDNET_NS, a_synset),
                          data_source, weight, {'image_id': image_id}]
            all_edges.append(attr_wn_edge)

            # save wordnet data for an attribute
            wn2label[a_synset].add(a)
            wn2image[a_synset].add(image_id)
            a_pos=a_synset.split('.')[-2]

            all_attr_synsets.append(a_synset)

        # attribute node
        a_label, a_aliases=add_lowercase_labels([a])
        attr_node=[a_id, a_label, ','.join(a_aliases), a_pos, data_source, {'image_id': image_id}]
        all_nodes.append(attr_node)


    return all_nodes, all_edges, wn2label, wn2image, attr_id


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
weight="1.0"

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

attr_id=1

for counter, an_image in enumerate(images_data):    
    image_id=an_image['image_id']
    
    # RELATIONSHIPS
    all_nodes, all_edges, wn2label, wn2image = add_relationships_data(an_image['relationships'], 
                                                                                      an_image['image_id'], 
                                                                                      all_nodes, 
                                                                                      all_edges,
                                                                                      wn2label,
                                                                                      wn2image)
    

    # OBJECTS
    for obj in an_image['objects']:
            
        obj_id=create_uri(VG_NS, 'O' + str(obj['object_id']))
        label, aliases=add_lowercase_labels(obj['names'])
        synsets=obj['synsets']
        
        pos=''
        if len(synsets):
            pos=synsets[0].split('.')[-2]
            for s in synsets:
                obj_wn_edge=[obj_id, WORDNET_SENSE_REL, create_uri(WORDNET_NS, s), data_source, weight, {'image_id': image_id}]
                all_edges.append(obj_wn_edge)

                # save wordnet data for an attribute
                wn2label[s].add(label)
                wn2image[s].add(image_id)
                
                all_obj_synsets.append(s)

        obj_node=[obj_id, label, ','.join(aliases), pos, data_source, {'image_id': image_id}]
        all_nodes.append(obj_node)

        # ATTRIBUTES
        attrs=obj['attributes'] if 'attributes' in obj else []
        all_nodes, all_edges, wn2label, wn2image, attr_id = add_attr_data(attrs, 
                                                                          image_id, 
                                                                          all_nodes, 
                                                                          all_edges, 
                                                                          wn2label, 
                                                                          wn2image, 
                                                                          attr_id)

print('number of synsets', len(set(all_rel_synsets) | set(all_attr_synsets) | set(all_obj_synsets)))

#### Add the synset data to the nodes.csv file ####

for synset, labels in wn2label.items():
    label, aliases=add_lowercase_labels(labels)
    pos=synset.split('.')[-2]
    images=list(wn2image[synset])
    wn_node=[create_uri(WORDNET_NS, synset), label, ','.join(aliases), pos, data_source, {'image_ids': images}]
    all_nodes.append(wn_node)

print('num nodes before adding geo data', len(all_nodes))
print('num edges before adding geo data', len(all_edges))

#### Add bounding box data from the region_graphs.json file ####

with open(vg_regions_path, 'r') as f:
    regions_data=json.load(f)

for image_data in regions_data:
    for region in image_data['regions']:
        image_id=create_uri(VG_NS, 'I' + str(region['image_id']))
        #bb_id=create_uri(VG_NS, 'B' + str(region['region_id']))
        
        #bb_image_edge=[bb_id, INIMAGE_REL, image_id, data_source, weight, {'image_id': region['image_id']}]
        #all_edges.append(bb_image_edge)
        
        for rel in region['relationships']:
            rel_id=create_uri(VG_NS, 'R' + str(rel['relationship_id']))
            rel_img_edge=[rel_id, INIMAGE_REL, image_id, data_source, weight, {}]
            all_edges.append(rel_img_edge)
            
        for obj in region['objects']:
            obj_id=create_uri(VG_NS, 'O' + str(obj['object_id']))
            obj_img_edge=[obj_id, INIMAGE_REL, image_id, data_source, weight, {}]
            all_edges.append(obj_img_edge)
            
        #bb_node=[bb_id, '', '', '', data_source, {'image_id': region['image_id'], 'sentence': region['phrase']}]
        #all_nodes.append(bb_node)
    image_node=[image_id, '', '', '', data_source, {}]
    all_nodes.append(image_node)

print('num nodes after adding geo data', len(all_nodes))
print('num edges after adding geo data', len(all_edges))

nodes_df=pd.DataFrame(all_nodes, columns = NODE_COLS)

# Drop duplicates
nodes_df.drop_duplicates(subset ="id",
                        keep = 'first', inplace = True)

print('combined nodes after deduplication:', len(nodes_df))

nodes_df.sort_values('id').to_csv(nodes_file, index=False, sep='\t')

edges_df = pd.DataFrame(all_edges, columns = EDGE_COLS)
# Drop duplicates
edges_df.drop_duplicates(subset =["subject", "predicate", "object"],
                             keep = 'first', inplace = True)

print('combined edges after deduplication:', len(edges_df))
edges_df.sort_values(by=['subject', 'predicate','object']).to_csv(edges_file, index=False, sep='\t')
