import os
import gzip
import faiss
from tqdm import tqdm
from lxml import etree
import numpy as np

########    ########    ########    ########    ########    ########    ############
# Wrapper for 'Evaluation between Bert text and USF-FAN.ipynb'
# This scrpit has the same code with 'Evaluation between Bert text and USF-FAN.ipynb'
########    ########    ########    ########    ########    ########    ############

"""
Class for data preparation
"""
class DataLoader():
    def __init__(self,cue_target,cskg_connected,bert_embs):
        """
        Parameters for invoking the notebook
        cue_target : a xml file contains the grounding truth of USF-FAN dataset
        bert_embs: a gzip(tsv) file contains the raw text's embedding information
        cskg_connected: a tsv file contains the raw cskg entity information
        embedding_folder: a folder contains all of the embedding cskg gz files
        MAP_res: a json file contains the MAP result for each cskg embedding gz file
        """
        self.cue_target = cue_target # '../input/cue-target.xml'
        self.bert_embs = bert_embs # '../input/bert-nli-large-embeddings.tsv.gz'
        self.cskg_connected = cskg_connected # '../input/cskg_connected.tsv'
        self.actual_max_num = 0

    ######################## USF-FAN loading #####################################
    def xml_load(self):   # cue-target.xml'
        cue_target = self.cue_target
        tree = etree.parse(cue_target)
        root = tree.getroot()
        # create a dict to store ground truth sets, 
        # example : `p={'car': ['wheel', 'driver', ...], 'book`: [...]}`
        ground_truth = {}
        for cue_ele in root:
            key = cue_ele.get('word').lower()
            ground_truth[key] = []
            for word_ele in cue_ele:
                ground_truth[key].append(word_ele.get('word').lower())
                
        # get the max_num of atcual items
        for items in ground_truth.values():
            if self.actual_max_num < len(items):
                self.actual_max_num = len(items)
                
        return ground_truth

    ######################## BERT large text loading #####################################
    def bert_load(self,file_length=2161049): # '../input/bert-nli-large-embeddings.tsv.gz'
        bert_embs = self.bert_embs

        text_embed_dict= {}
        with gzip.open(bert_embs, 'rb') as f:
            for line in tqdm(f,total=file_length): # prerun it to get the total number 2161049
                line = line.decode()
                node,prop,value = line.split('\t')
                value = value.split(',')
                if node == 'node': # ignore the first line 
                    continue 
                embedding = [ float(i) for i in value]
                text_embed_dict[node] = embedding 
                
        return text_embed_dict

    ######################## CSKG lable loading #####################################
    def cskg_load(self,file_length=6003238): # cskg_connected.tsv
        cskg_connected = self.cskg_connected
        # create a dict to store cskg data set   label: node_list
        # example : `p={'turtle':  ['Q1705322', '/c/en/turtle', ...], 'book`: [...]}`
        cskg_index_dict = {}
    
        # create an inverted index to record lbl and node mapping
        # example: p={'Q1705322': 'turtle', '/c/en/turtle': 'turtle'}
        lbl_node_inv_index = {}

        with open(cskg_connected) as f:
            for line in tqdm(f,total=file_length):
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
                    
                    lbl_node_inv_index[node1_id] = node1_lbl
                    lbl_node_inv_index[node2_id] = node2_lbl
                    
        # convert set to list
        for k in cskg_index_dict:
            cskg_index_dict[k] = list(cskg_index_dict[k])

        return cskg_index_dict,lbl_node_inv_index
    
    ################# Util ###################################################
    #Umapping txt to cskg => get common label's embeddings
    def map_txt_cskg(self,text_embed_dict,lbl_node_inv_index):    
        txt_lbl_emb_dict = {}
        text_num = len(text_embed_dict)
        for node in tqdm(text_embed_dict.keys(),total=text_num): 
            if node in lbl_node_inv_index:
                label = lbl_node_inv_index[node]
                txt_lbl_emb_dict[label] = text_embed_dict[node]
            
        return txt_lbl_emb_dict

"""
Class for data processing
"""
class DataProcesser():
    def __init__(self):
        pass
    
    def build_fassi_index(self,txt_lbl_emb_dict):
        # txt_lbl_emb_dict is a dictionary which key is the node label and value is lable's embedding  
        label_dict = {}         # build a entity label-index bi dictionary
        entity_embeddings = []  # all the embeddings 
        index = 0
        for key,value in txt_lbl_emb_dict.items():
            label_dict[index] = key
            label_dict[key] = index
            index += 1    
            entity_embeddings.append(value)

        # entity_embeddings => matrix  X contains  all labels' embeddings 
        X = np.array(entity_embeddings).astype(np.float32)   # float32
        dimension = X.shape[1]

        # build index (METRIC_INNER_PRODUCT => cos )
        vec_index = faiss.index_factory(dimension, "Flat", faiss.METRIC_INNER_PRODUCT)
        # normalize all vectors in order to get cos sim 
        faiss.normalize_L2(X)  
        # add vectors to index
        vec_index.add(X) 
        
        return vec_index,label_dict,X

    def create_queryset(self,USF_FAN_dict,label_dict,txt_lbl_emb_dict):
        query_ent_vecs = []
        query_ent_dict = {}
        miss_concept = 0
        miss_concept_list = []
        
        for key in USF_FAN_dict:
            if key in txt_lbl_emb_dict:
                query_ent_dict[len(query_ent_vecs)] = key
                query_ent_dict[key] = len(query_ent_vecs)
                query_ent_vecs.append(txt_lbl_emb_dict[key])
            else:
                miss_concept_list.append(key)
                miss_concept+=1
                
        print(f'match label num from cskg and USF-FAN: {len(query_ent_vecs)}')
        print(f'miss label num from cskg and USF-FAN: {miss_concept}, they are {miss_concept_list}')
        query_ent_mat = np.array(query_ent_vecs).astype(np.float32)
        faiss.normalize_L2(query_ent_mat) 
        return query_ent_mat,query_ent_dict
   
    def neighbor_searching(self,vec_index,query_ent_mat,query_ent_dict,label_dict,k,fix_num):
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
    
    ### Evaluation
    def apk(self,actual, predicted, k):   
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

        
    def map_at_k(self,pre_dict,grouding_dict,k):
        MAP = 0 
        set_size = len(pre_dict) 

        # cal ap
        for label in pre_dict:
            predicted = pre_dict.get(label,[])
            actual = grouding_dict.get(label,[])
            ap = self.apk(actual, predicted, k)
            MAP+=ap

        return MAP/set_size
    
    def reci_rank(self,actual, predicted):
        # The inverse of the ranking of the first correct answer
        # keep both predicted's order and actual's order
        for i in  predicted:
            if i in actual:
                return 1/(actual.index(i)+1)
            
        return 0 # no match     
    
    def MPR(self,pre_dict,grouding_dict):
        MPR = 0
        set_size = len(pre_dict)
        
        for label in pre_dict:
            predicted = pre_dict.get(label,[])
            actual = grouding_dict.get(label,[])
            rr = self.reci_rank(actual, predicted)
            MPR+=rr
            
        return MPR/set_size


if __name__ == "__main__":

    cue_target = 'input/cue-target.xml'
    cskg_connected = 'input/cskg_connected.tsv'
    bert_embs = 'input/bert-nli-large-embeddings.tsv.gz'

    ## Data preparation
    dataLoader = DataLoader(cue_target,cskg_connected,bert_embs)

    # 1. loda USF_FAN dataset
    print("Loda USF_FAN dataset...")
    USF_FAN_dict = dataLoader.xml_load()
    print(f"USF_FAN_dict['clock']: {USF_FAN_dict['clock']}")

    # 2. load cskg dataset => here we only need its mapping of label and node
    print("Load CSKG label and node data...")
    cskg_index_dict,lbl_node_inv_index = dataLoader.cskg_load() 

    # 3. text dataset load and mapping with cskg dataset
    print("Loda BERT-NLI-large text embeddings...")
    txt_embed_dict = dataLoader.bert_load()
    txt_lbl_emb_dict = dataLoader.map_txt_cskg(txt_embed_dict,lbl_node_inv_index)
    print(f"large text data nodes number: {len(txt_embed_dict)}")
    print(f"matched node number with cskg: {len(txt_lbl_emb_dict)}")

    ## Data processing
    dataProcesser = DataProcesser()

    # 4. create vector index for bert-nli-large-embeddings by using fassi
    print('Create vector index...')
    vec_index,label_dict,X = dataProcesser.build_fassi_index(txt_lbl_emb_dict)

    # 5. create query matrix bert-nli-large-embeddings 
    print('Create query entity matrix...')
    query_ent_mat,query_ent_dict = dataProcesser.create_queryset(USF_FAN_dict,label_dict,txt_lbl_emb_dict)

    # 6. search neighbors for matching items
    print('Do searching for query entity matrix...')
    fix_num = dataLoader.actual_max_num # get the largest community num.
    # @1 neighbors
    k = 1
    neigbors_dict = dataProcesser.neighbor_searching(vec_index,query_ent_mat,query_ent_dict,label_dict,k,fix_num)
    print(neigbors_dict['give up'], USF_FAN_dict['give up'])

    # 7. calculate the map for predicted neigbors (compared to USF-FAN)
    MAP = dataProcesser.map_at_k(neigbors_dict,USF_FAN_dict,k)
    print(f"MAP@{k} for predicted neigbors: {MAP}")

    #8. calculate the mrr for predicted neigbors (compared to USF-FAN)
    MPR = dataProcesser.MPR(neigbors_dict,USF_FAN_dict)
    print(f"MPR for predicted neigbors: {MPR}")

    # get all result
    ks = [1,2,3,5,10]
    for k in ks:
        neigbors_dict = dataProcesser.neighbor_searching(vec_index,query_ent_mat,query_ent_dict,label_dict,k,fix_num)
        MAP = dataProcesser.map_at_k(neigbors_dict,USF_FAN_dict,k)
        MPR = dataProcesser.MPR(neigbors_dict,USF_FAN_dict)
        print(f"MAP@{k} for predicted neigbors: {MAP}")
        print(f"MPR@{k} for predicted neigbors: {MPR}") 
        print('\n')