import json
import copy
import faiss
import gzip
import os
import numpy as np
from lxml import etree
from tqdm import tqdm

########    ########    ########    ########    ########    ########    ############
# Wrapper for 'Evaluation between CSKG and USF-FAN.ipynb'
# This scrpit has the same code with 'Evaluation between CSKG and USF-FAN.ipynb'
# Execute may take 16 hours to genereate all results 48*5*5
########    ########    ########    ########    ########    ########    ############



######################### Util ################################
def dict_to_json(dict_,output_file):
    with open(output_file,'w') as f:
        json.dump(dict_,f)
         
def get_file_path(embedding_folder):
    gz_list = []
    for gz_file in os.listdir(embedding_folder):
        file_path = os.path.join(embedding_folder, gz_file)
        gz_list.append(file_path)   
    return gz_list
    
def get_max_num(ground_truth_dict):
    max_num = 0
    for label in ground_truth_dict:
         if max_num < len(ground_truth_dict[label]):
                max_num = len(ground_truth_dict[label])
    return max_num       
####################################################################

######################### Load Data ################################
def xml_load(input_file):   # cue-target.xml'
    tree = etree.parse(input_file)
    root = tree.getroot()
    # create a dict to store ground truth sets, 
    # example : `p={'car': ['wheel', 'driver', ...], 'book`: [...]}`
    ground_truth = {}
    for cue_ele in root:
        key = cue_ele.get('word').lower()
        ground_truth[key] = []
        for word_ele in cue_ele:
            ground_truth[key].append(word_ele.get('word').lower())
    return ground_truth

def create_cskg_index(tsv_file): # cskg_connected.tsv
    cskg_index_dict = {}
    #  create a dict to store cskg data set   label: node_list
    #  example : `p = {'turtle':  ['Q1705322', '/c/en/turtle', ...], 'book': [...]}`
    with open(tsv_file) as f:
        for line in f:
            content = line.split('\t')
            if content[0]!='id': # ignore the first time 
                node1_id = content[1]
                node2_id = content[3]
                node1_lbl = content[4]
                node2_lbl = content[5]
                cskg_index_dict[node1_lbl] = cskg_index_dict.get(node1_lbl,set())
                cskg_index_dict[node1_lbl].add(node1_id)
                cskg_index_dict[node2_lbl] = cskg_index_dict.get(node2_lbl,set())
                cskg_index_dict[node2_lbl].add(node2_id)
                
    # convert set to list
    for k in cskg_index_dict:
        cskg_index_dict[k] = list(cskg_index_dict[k])

    return cskg_index_dict

def load_ent_embeddings(input_file):
    # input file folder path :/nas/home/binzhang/backup_data/embeddings 
    #  create a dict to store cskg embeddings   node: embedding example: 
    # '/c/en/turtle': [0.01,0.02....]

    ix_node_dict = {} # { node_index: node_name,... node_name:node_index... }
    node_embedding_dict = {} # {node_name:embedding, ....}
    with gzip.open(input_file,'rt') as f:
        for index,line in enumerate(f):
            line = line.split('\t')
            entity_name = line[0]
            entity_vec =  [ float(i) for i in line[1:]]
            ix_node_dict[entity_name] = index
            ix_node_dict[index] = entity_name
            node_embedding_dict[entity_name] = entity_vec
    
    return ix_node_dict,node_embedding_dict

#######################################################################


######################### Process Data ################################

def cal_avg_embeddings(node_embedding_dict,cskg_index_dict):
    # node_embedding_dict's key is node's name (example: '/c/en/joke') 
    # and the value is the embedding vectors
    # example: '/c/en/turtle': [0.01,0.02...]

    # cskg_index_dict's key is the label for a node , value is a list recording the node's name
    # example : 'joke': ['/c/en/joke', '/c/en/joke/n', '/c/en/joke/n/wn/act',...]
    avg_embeddings = {}
    for label in cskg_index_dict:
        entity_names = cskg_index_dict[label]
        size = len(entity_names)
        sum_embedding =  node_embedding_dict[entity_names[0]]
        for entity in entity_names[1:]:
            embedding = node_embedding_dict[entity] # embeddings list 
            sum_embedding = list(map(lambda x,y : x+y ,sum_embedding,embedding))
            
        avg_emb = [i/size for i in sum_embedding]
        avg_embeddings[label] = avg_emb
        
    return avg_embeddings

def build_fassi_index(avg_embeddings):
    # avg_embeddings is a dictionary which key is the node label and value is lable's embedding
    
    label_dict = {}         # build a entity label-index bi dictionary
    entity_embeddings = []  # all the embeddings 

    index = 0
    for key,value in avg_embeddings.items():
        label_dict[index] = key
        label_dict[key] = index
        index+=1    
        entity_embeddings.append(value)

    # entity_embeddings => matrix  X contains  all labels' embeddings 
    X = np.array(entity_embeddings).astype(np.float32) # float32
    dimension = X.shape[1]

    # build index (METRIC_INNER_PRODUCT => cos )
    vec_index = faiss.index_factory(dimension, "Flat", faiss.METRIC_INNER_PRODUCT)
    # # normalize all vectors in order to get cos sim 
    faiss.normalize_L2(X)  
    # add vectors to inde 
    vec_index.add(X) 
    
    return vec_index,label_dict

def create_queryset(ground_truth_dict,label_dict,avg_embeddings):
    query_ent_vecs = []
    query_ent_dict = {}
    miss_concept = 0
    miss_concept_list = []
    
    for key in ground_truth_dict:
        if key in label_dict: 
            query_ent_dict[len(query_ent_vecs)] = key
            query_ent_dict[key] = len(query_ent_vecs)
            query_ent_vecs.append(avg_embeddings[key])
        else:
            miss_concept_list.append(key)
            miss_concept+=1

    query_ent_mat = np.array(query_ent_vecs).astype(np.float32)
    faiss.normalize_L2(query_ent_mat) 
    
    return query_ent_mat,query_ent_dict

def neighbor_searching(vec_index,query_ent_mat,query_ent_dict,label_dict,k,fix_num):
    # k = times of items => k = 1 ,reterice @1 items k = 3 ,reterice @3 items 

    neigh_num = k*fix_num
    cos_sim, index = vec_index.search(query_ent_mat, neigh_num)     # both of them are matrices

    neighbors_dict = {}
    for ix,neighbors in enumerate(index):
        query_item = query_ent_dict[ix]
        tmp_list = []
        for id_ in neighbors:
            tmp_list.append(label_dict[id_])            # ix refers to the label's index 

        neighbors_dict[query_item] = tmp_list

    return neighbors_dict   


## evaluation metric
def apk(actual, predicted, k):   
    # keep predicted's order igonore actual's order
    if len(predicted)>k*len(actual):
        predicted = predicted[:k]
    ap = 0.0
    num_hits = 0.0
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            ap += num_hits / (i+1.0)

    if num_hits == 0:  # no match from predict and actual 
        return 0.0
    else:
        return ap / num_hits
    
def map_at_k(pre_dict,grouding_dict,k):
    MAP = 0 
    set_size = len(pre_dict) 

    # cal ap
    for label in pre_dict:
        predicted = pre_dict.get(label,[])
        actual = grouding_dict.get(label,[])
        ap = apk(actual, predicted, k)
        MAP+=ap

    return MAP/set_size

    
#######################################################################

def reci_rank(actual, predicted):
    # The inverse of the ranking of the first correct answer
    # keep both predicted's order and actual's order
    for i in  predicted:
        if i in actual:
            return 1/(actual.index(i)+1)

    return 0 # no match     

def MPR(pre_dict,grouding_dict):
    MPR = 0
    set_size = len(pre_dict)

    for label in pre_dict:
        predicted = pre_dict.get(label,[])
        actual = grouding_dict.get(label,[])
        rr = reci_rank(actual, predicted)
        MPR+=rr

    return MPR/set_size

if __name__ == "__main__":
    cue_target = 'input/cue-target.xml'
    cskg_connected = 'input/cskg_connected.tsv'
    embedding_folder = 'output/embeddings'
    evaluation_res = 'output/evaluation_res'

    # 1. load  USF-FAN dataset 
    ground_truth_dict = xml_load(cue_target)
    max_num = get_max_num(ground_truth_dict)

    # 2. load CSKG data
    # construct an index of CSKG from label to node id  result: cskg_index_dict
    cskg_index_dict = create_cskg_index(cskg_connected)

    #3. get all possible possible embeddings from  embedding_folder
    gz_list = get_file_path(embedding_folder)

    # If we want to get all possible results, you can and make a for loop to execute
    K = [1,2,3,5,10]
    for k in tqdm(K,total=len(K)):
        MAPs,MPRs  = {},{} 
        # assign output file path
        map_out =  f'{evaluation_res}/MAP@{k}.json'
        mpr_out =  f'{evaluation_res}/MPR@{k}.json'
        
        for ent_embedding_path in tqdm(gz_list,total=len(gz_list)): 
            # obtain the embeddings for all concepts 
            ix_node_dict,node_embedding_dict = load_ent_embeddings(ent_embedding_path)        
            # compute an average embedding. 
            avg_embeddings = cal_avg_embeddings(node_embedding_dict,cskg_index_dict)
            # faiss: create vector index
            vec_index,label_dict= build_fassi_index(avg_embeddings)
            # create query sets based on grounding truth(USF-FAN)
            query_ent_mat,query_ent_dict = create_queryset(ground_truth_dict,label_dict,avg_embeddings)
            
            # do neighbor searching
            neighbors_dict = neighbor_searching(vec_index,query_ent_mat,query_ent_dict,label_dict,k,max_num)
                
            #calculate metrics
            MAP = map_at_k(neighbors_dict,ground_truth_dict,k)
            mpr = MPR(neighbors_dict,ground_truth_dict)
            emb_key = ent_embedding_path.split('/')[-1]
            MAPs[emb_key] = MAP
            MPRs[emb_key] = mpr
            
        dict_to_json(MAPs,map_out)
        dict_to_json(MPRs,mpr_out)  
        print(f'MAP@{k} has been calculated for all embeddings')
        print(f'MPR@{k} has been calculated for all embeddings')
        print()    