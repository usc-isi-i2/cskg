import gzip
from lxml import etree
from tqdm import tqdm
import faiss
import json
import numpy as np
import rltk
from itertools import islice
from sklearn.metrics import ndcg_score
import time

"""
A set of util functions for CSKG entity analysis
"""

def xml_load(input_file):   
    """
    Load the data from cue-target.xml (A cue to target file, the cue are sorted in alphabetical 
    order and within a cue the targets are in decreasing order of similarity.)

    Parameters
    ----------
    input_file : str
        The file path of cue-target.xml

    Returns
    -------
    ground_truth: dict
        A dictionary whose key is a cue's label, value is a list containing cue's similar targets 
        in decreasing order of similarity
        example for a key: 'turtle': ['slow','shell','tortoise','animal',...]
    """
    tree = etree.parse(input_file)
    root = tree.getroot()
    # create a dict to store ground truth sets, 
    # example : `p={'car': ['wheel', 'driver', ...], 'book`: [...]}`
    USF_FAN_dict = {}
    for cue_ele in root:
        key = cue_ele.get('word').lower()
        USF_FAN_dict[key] = []
        for word_ele in cue_ele:
            USF_FAN_dict[key].append(word_ele.get('word').lower())
    return USF_FAN_dict

def cskg_load(input_file): 
    """
    Load the data from cskg_connected.tsv (A tsv file contains the raw cskg's information, 
    each line's format is: id  node1 relation  node2  node1;label  node2;label  relation;label  
    relation;dimension  source  sentence)

    Parameters
    ----------
    input_file : str
        The file path of cskg_connected.tsv

    Returns
    -------
    CSKG_label_dict: dict
        A dictionary whose key is the label of the node, value is a list of node IDs, 
        whode node's label is the corresponding key.
        example for a key: 'turtle': ['Q1705322', '/c/en/turtle', ...], 'book`: [ 'Q997698','/c/en/book/v', ...]
    CSKG_inv_dict: dict
        A inverted index dictionary recording the correspondence between the ID and label of each node.
        The key is the node ID, the value is the node's label corresponding to the ID
        example for some keys: 'Q1705322':'turtle', '/c/en/turtle', 'Q997698':'book'
    """
    CSKG_label_dict = {}
    CSKG_inv_dict = {}
    with open(input_file) as f:
        for line in islice(f, 1, None):  # ignore first line(header) 
            content = line.split('\t')
            node1_id = content[1]  # node 1 id
            node2_id = content[3]  # node 2 id
            node1_lbl = content[4] # node 1 label
            node2_lbl = content[5] # node 2 label 
            # There might exists labels with multiple terms => CSKG_label_dict['allay|ease|relieve|still'], 
            # in this case we only use first term "allay"
            node1_lbl = node1_lbl.split('|')[0]
            node2_lbl = node2_lbl.split('|')[0]

            CSKG_label_dict[node1_lbl] = CSKG_label_dict.get(node1_lbl,set())
            CSKG_label_dict[node1_lbl].add(node1_id)
            CSKG_label_dict[node2_lbl] = CSKG_label_dict.get(node2_lbl,set())
            CSKG_label_dict[node2_lbl].add(node2_id)
            CSKG_inv_dict[node1_id] = node1_lbl
            CSKG_inv_dict[node2_id] = node2_lbl
   
    for k in CSKG_label_dict.keys():
        # convert set to list
        CSKG_label_dict[k] = list(CSKG_label_dict[k])

    return CSKG_label_dict,CSKG_inv_dict

def get_ground_truth(USF_FAN_dict,CSKG_label_dict):
    """
    Generate the ground turth for following analysi

    Parameters
    ----------
    USF_FAN_dict : dict
        A dictionary whose key is a cue's label, value is a list containing cue's similar targets 
        in decreasing order of similarity
        example for a key: 'turtle': ['slow','shell','tortoise','animal',...]

    CSKG_label_dict: dict:
        A dictionary whose key is the label of the node, value is a list of node IDs, 
        whode node's label is the corresponding key.

    Returns
    -------
    ground_truth: dict
        A dictionary whose key is both in USF_FAN and CSKG, value the same value as the USF_FAN_dict for the cue
    """
    ground_truth = {}
    for cue in USF_FAN_dict:
        if cue in CSKG_label_dict: 
            ground_truth[cue] = USF_FAN_dict[cue]  
    return ground_truth

def txt_emb_load(input_file):
    """
    Load the data from bert-nli-large-embeddings.tsv.gz. (A tsv file in .gz format contains the text embedding for 
    nodes.  each line's format is: node   property   value)

    Mention
    ----------
    Due to the large size of thie file, we first use `wc -l bert-nli-large-embeddings.tsv` to get the # of lines 
    of this file and use tqdm module to show processing
    
    Parameters
    ----------
    input_file : str
        The file path of bert-nli-large-embeddings.tsv.gz.

    Returns
    -------
    text_embed_dict: dict
        A dictionary whose key is the Node ID, value is the text embeddings for such node.
        example: 'wn:zidovudine.n.01':[-0.17918809,-0.13626638,0.4172361,-0.5083385,-0.22449987,...]
    """
    file_length = 2161049 
    text_embed_dict = {}
    with gzip.open(input_file, 'rb') as f:
        for line in tqdm(islice(f, 1, None),total=file_length-1,ncols=80):  # ignore the header 
            line = line.decode()

            node,prop,value = line.split('\t')
            value = value.split(',')
            embedding = [ float(i) for i in value]
            text_embed_dict[node] = embedding 
            
    return text_embed_dict

def graph_emb_load(input_file):
    """
    Load the data from trans_log_dot_0.1.tsv.gz. (A tsv file in .gz format contains the graph embedding for 
    nodes.  each line includes node id and embeddings.
    
    Parameters
    ----------
    input_file : str
        The file path of trans_log_dot_0.1.tsv.gz

    Returns
    -------
    graph_embed_dict: dict
        A dictionary whose key is the Node ID, value is the graph embeddings for such node.
        example: 'wn:zidovudine.n.01': [-0.23314859,-0.35881421,0.331990957,-0.112634301,0.047500141,0.567292035,...]
    """
    graph_embed_dict = {}
    with gzip.open(input_file,'rt') as f:
        for index,line in enumerate(f):
            line = line.split('\t')
            node_id = line[0]
            value =  [ float(i) for i in line[1:]]
            graph_embed_dict[node_id] = value
    
    return graph_embed_dict

def get_label_emb(node_emb_dict,CSKG_label_dict):
    """
    Calculate the average embeddings for  labels on CSKG

    Parameters
    ----------
    node_emb_dict : dict
        A dictionary whose key is the Node ID, value is the graph embeddings for such node.

    CSKG_label_dict: dict
        A dictionary whose key is the label of the node, value is a list of node IDs.

    Returns
    -------
    label_emb_dict: dict
        A dictionary whose key is the Node label, value is the average embedding for such node. 
        example: 'turtle': [-0.23314859,-0.35881421,0.331990957,-0.112634301,0.047500141,0.567292035,...]
    """
    avg_embeddings = {}
    for label in CSKG_label_dict:
        list_node = CSKG_label_dict[label]
        num_node = len(list_node)
        sum_embedding =  node_emb_dict.get(list_node[0],[]) # get first node's embedding as the initial embedding
        if not sum_embedding: # no embdding for such node id:
            continue
        for node in list_node[1:]:
            embedding = node_emb_dict.get(node,[])
            if not embedding:
                num_node-=1
                continue
            sum_embedding = list(map(lambda x,y : x+y ,sum_embedding,embedding))       
        avg_emb = [i/num_node for i in sum_embedding]
        avg_embeddings[label] = avg_emb
        
    return avg_embeddings

def build_index(label_emb):
    """
    Build a Faiss index for label embddings for future neighbor searching, here since cosine similarity will be 
    adapted, FlatIndex is used

    Parameters
    ----------
    label_emb : dict
        A dictionary whose key is the Node label, value is the average embeddings for such node.

    Returns
    -------
    vec_index: faiss.swigfaiss_avx2.IndexFlat
        A IndexFlat that keeps the index for the label embeddings. The index number of each label is determined according 
        to the order of adding, for example, the first index is 0.
    
    label_ix_dict: dict
        A dictionary whose key is the index number, value is the label. This dictionary is aimed at recording each label's 
        order for future mapping.
    """   
    label_ix_dict = {}         
    node_embeddings = []    

    index = 0
    for key,value in label_emb.items():
        label_ix_dict[index] = key
        index+=1    
        node_embeddings.append(value)

    # entity_embeddings => matrix  X contains  all labels' embeddings 
    X = np.array(node_embeddings).astype(np.float32) # float32
    dimension = X.shape[1]
    # build index (METRIC_INNER_PRODUCT => cos )
    vec_index = faiss.index_factory(dimension, "Flat", faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(X)  # normalize all vectors in order to get cos sim 
    # add vectors to inde 
    vec_index.add(X) 

    return vec_index,label_ix_dict

def create_queryset(ground_truth,CSKG_label_dict,label_emb_dict):
    """
    Create queryset for CSKG's labels according to USF_FAN's cues 

    Parameters
    ----------
    ground_truth : dict
        A dictionary whose key is a cue's label, value is a list containing cue's similar targets 
        in decreasing order of similarity
    
    CSKG_label_dict: dict
        A dictionary whose key is the label of the node, value is a list of node IDs, 
        whode node's label is the corresponding key.

    label_emb_dict: dict
        A dictionary whose key is the node's label, value is the average embedding for such node. 

    Returns
    -------
    query_dict:  dict
        A dictionary whose key is both in ground_truth and CSKG, value is the graph embedding value 
        for labels on CSKG nodes

    """ 
    query_dict = {}
    for cue in ground_truth:
        # convert the label's embedding to the same foramt as faiss for future searching
        label_embedding = label_emb_dict[cue]
        query_mat =  np.array(label_embedding).reshape((1,-1)).astype(np.float32)
        faiss.normalize_L2(query_mat) 
        query_dict[cue] = query_mat
    return query_dict

def get_label_neighbor(label_vec,vec_index,label_ix_dict,X,include=False):
    """
    Search neighbors for a label according to cosine similarity among labels' embedding

    Parameters
    ----------
    
    label_vec: np.ndarray
        a normalized vector whose shape is 1 x n  , this vector is the embedding value for a label

    vec_index: faiss.swigfaiss_avx2.IndexFlat
        A IndexFlat that keeps the index for the label embeddings. The index number of each label is 
        determined according to the order of adding, for example, the first index is 0.
    
    label_ix_dict: dict
        A dictionary whose key is the index number, value is the label. This dictionary is aimed at 
        recording each label's order for future mapping.
    
    X: int
        how many similar targets should be returned 
    
    include: boolean [default:False]
        When do searching, whether include the label itself, if set True, then the first neighbor is the label itself
    
    Returns
    -------
    neighbors: list 
        A list containing the label's similar targets in decreasing order of cosine similarity.
        each item in the list is a tuple, format is (target, similarity to the label)
    """
    
    neighbors = []
    cos_sim, index = vec_index.search(label_vec, X+1) 
    for ix in index[0][:]: # ignore itself  index is 1x(X+1) so need to use index[0] get the list
        target = label_ix_dict[ix]
        neighbors.append(target)

    if include:
        relevance = cos_sim[0][:X]
        neighbors = list(zip(neighbors[:X],relevance))
        return neighbors
    else:
        relevance = cos_sim[0][1:X+1]
        neighbors = list(zip(neighbors[1:],relevance))
        return neighbors
     
def neighbor_search(query_dict,ground_truth,vec_index,label_ix_dict,X):
    """
    Search neighbors for queryset according to cosine similarity among labels' embedding

    Parameters
    ----------
    query_dict : dict
        A dictionary whose key is both in ground_truth and CSKG, value is the embedding 
        value according to CSKG
    
    ground_truth: dict
        A dictionary whose key is a cue's label, value is a list containing cue's similar targets 
        in decreasing order of similarity

    vec_index: faiss.swigfaiss_avx2.IndexFlat
        A IndexFlat that keeps the index for the label embeddings. The index number of each label is 
        determined according to the order of adding, for example, the first index is 0.
    
    label_ix_dict: dict
        A dictionary whose key is the index number, value is the label. This dictionary is aimed at recording each label's 
        order for future mapping.
    
    X: int
        Indicates how many times the data is extracted from CSKG according to the ground truth's label's targets number
        example: if we set X = 1  , and for a cue 'turtle' in USF_FAN_dict, there exists 18 similar targets for this cue,
        then we will search 18 similar labels for 'turtle' in CSKG  

    Returns
    -------
    neighbor_dict: dict
        A dictionary whose key is a label in CSKG, value is a list containing the label's similar targets 
        in decreasing order of cosine similarity, each item in the list is a tuple, first item is the similar 
        target, and second one is the similarity to the label.
        example:
        {'a': [('s', 0.9048489),('more', 0.88388747),('c', 0.8800387)...]...}
    """ 
    neighbor_dict = {}
    for query_label in tqdm(query_dict,total=len(query_dict),ncols=80):    
        #number of  target for this label/cue in ground turth
        target_num = len(ground_truth[query_label])
        # query label's embedding on CSKG
        label_vec = query_dict[query_label]
        # searched targets
        neighbors = get_label_neighbor(label_vec,vec_index,label_ix_dict,X*target_num)
        neighbor_dict[query_label] = neighbors
    
    return neighbor_dict

def get_pred_dict(neighbor_dict):
    """
    Generate the same foramt with ground turth for future analysis 

    Parameters
    ----------
    neighbor_dict : dict
        A dictionary whose key is a label in CSKG, value is a list containing the label's similar targets 
        in decreasing order of cosine similarity, each item in the list is a tuple, first item is the similar 
        target, and second one is the similarity to the label.
        example:
        {'a': [('s', 0.9048489),('more', 0.88388747),('c', 0.8800387)...]...}
    
    Rreturn 
    ----------
    pred_dict: dict
        A dictionary whose key is a cue's label, value is a list containing cue's similar targets 
        in decreasing order of similarity generated by faiss neighbor searching
    """
    pred_dict = {}
    for label in neighbor_dict:
        pred_dict[label] = [i[0] for i in neighbor_dict[label]]
    return pred_dict


def cal_hits(ground_truth,pred_dict,level='macro'):
    """
    Create Hit@X for CSKG labels compared with ground_truth

    Parameters
    ----------
    ground_truth : dict
        A dictionary whose key is a cue's label, value is a list containing cue's similar targets 
        in decreasing order of similarity 

    pred_dict: dict
        A dictionary whose key is a label in CSKG, value is a list containing the label's 
        similar targets in decreasing order of cosine similarity. (faiss)

    level: str [default:'macro']
        Use micro or macro ways to calculate hits
        if micro, hits = average( (# of hits for a single label)/ (# of tragets in ground turth))
        if macro, hits =  sum(number of hits) / sum(ground_truth number)

    Returns
    -------
    Hit: float
        Hit socre: computes how many elements of a vector of rankings ranks make it 
        to the top n positions.
    """ 
    cue_num = len(ground_truth)
    hits = 0
    if level == 'micro':
        for label in pred_dict:
            pred_neighbors = set(pred_dict[label])
            truth_neighbos = set(ground_truth[label])
            true_hits = len(truth_neighbos) 
            tmp_hits = len(pred_neighbors&truth_neighbos)
            hits += tmp_hits/true_hits
        
        return hits/cue_num
    else:  # 'macro'
        sum_num = 0
        for label in pred_dict:
            pred_neighbors = set(pred_dict[label])
            truth_neighbos = set(ground_truth[label])
            tmp_hits = len(pred_neighbors&truth_neighbos)
            hits+=tmp_hits
            sum_num += len(truth_neighbos)
        return hits/sum_num

def cal_mrr(ground_truth,pred_dict):
    """
    Create MRR(mean reciprocal rank) for CSKG labels compared with ground_truth

    Parameters
    ----------
    ground_truth : dict
        A dictionary whose key is a cue's label, value is a list containing cue's similar targets 
        in decreasing order of similarity

    pred_dict:
        A dictionary whose key is a label in CSKG, value is a list containing the label's similar targets 
        in decreasing order of cosine similarity.

    Returns
    -------
    mrr : float
        nrr socre:  a measure to evaluate systems that return a ranked list of answers to queries
        Formula: Hit =  sum(  1 / position of rank1 for each label) / sum(ground_truth number)
    """ 
    sum_size = 0
    mrr = 0
    for label in pred_dict:
        rank1 = pred_dict[label][0]
        if rank1 in ground_truth[label]:
            rr = 1/ (ground_truth[label].index(rank1) + 1)  
        else:
            rr = 0
        mrr +=rr
        sum_size+=1
    
    return mrr/sum_size

def cal_map(ground_truth,pred_dict):
    """
    Create MAP(mean average precision) for CSKG labels compared with ground_truth

    Parameters
    ----------
    ground_truth : dict
        A dictionary whose key is a cue's label, value is a list containing cue's similar targets 
        in decreasing order of similarity

    pred_dict:
        A dictionary whose key is a label in CSKG, value is a list containing the label's similar targets 
        in decreasing order of cosine similarity.

    Returns
    -------
    MAP : float
        MAP socre:  Mean average precision for a set of queries is the mean of the average 
        precision scores for each query.
        Formula: Hit =  sum(AP)  / sum(ground_truth number)
    """ 
    def cal_ap(actual, predicted):   
        ap = 0   # keep predicted's order ignore actual's order
        num_hits = 0
        for i,p in enumerate(predicted):
            if p in actual:
                num_hits += 1
                ap += num_hits / (i+1)
                
        if num_hits == 0:  # no match from predict and actual 
            return 0
        else:
            return ap/num_hits

    sum_size = 0 
    MAP = 0
    for label in pred_dict:
        predicted = pred_dict[label]
        actual = ground_truth[label]
        ap = cal_ap(actual, predicted)
        MAP+=ap
        sum_size+=1

    return MAP/sum_size


def cal_ndcg(ground_truth,pred_dict):
    """
    Create NDCG(Normalized Discounted Cummulative Gain) for CSKG labels compared with ground_truth

    Parameters
    ----------
    ground_truth : dict
        A dictionary whose key is a cue's label, value is a list containing cue's similar targets 
        in decreasing order of similarity

    pred_dict:
        A dictionary whose key is a label in CSKG, value is a list containing the label's similar targets 
        in decreasing order of cosine similarity.


    Returns
    -------
    NDCG: : float in [0., 1.]
        The averaged NDCG scores for all labels.
    """
    def generate_y_true(targets):
        y_ture = []
        num = len(targets)
        return [ (i+1)/num  for i,_ in enumerate(targets)][::-1] # 
        # example: generate_y_true(['hellp','dsad','dasd']) => [1,0.6666,0.3333]

    NDCG = 0
    size = len(ground_truth)
    for cue in ground_truth:
        true_targets = ground_truth[cue]
        pred_targets = pred_dict[cue]
        pred_targets_size = len(pred_dict[cue])
        y_trues = generate_y_true(true_targets)
        y_scores = []

        for target in true_targets:
            if target in pred_targets:
                position = pred_targets.index(target) 
                relevance = (pred_targets_size - position )/ pred_targets_size
                y_scores.append(relevance)
            else:
                # if cannot find the target in so many candidates, then the relevance will be pretty small
                y_scores.append(0)  
        
        y_trues = np.array(y_trues).reshape((1,-1))
        y_scores = np.array(y_scores).reshape((1,-1))
        try:
            tmp_NDCG = ndcg_score(y_trues,y_scores)
        except:
            tmp_NDCG = 1 # when  y_trues = [[1]] y_scores = [[1]] , it will run an error
        NDCG+=tmp_NDCG
    
    NDCG = NDCG/size
    return NDCG


    # for label in neighbor_dict:
    #     true_neighbors = ground_truth[label]
    #     y_trues = generate_y_true(true_neighbors)

    #     y_scores = [0] * len(y_trues)
    #     pred_neighbors = neighbor_dict[label] # a list of tuples
    #     ## Idea: we find a target which both in ground turth and neighbor_dict to use this target's simialrity 
    #     ## as a base relevance 
    #     for index,(target,sim) in enumerate(pred_neighbors):
    #         if target in true_neighbors:
    #             position = true_neighbors.index(target)
    #             y_ture = y_trues[position]  
    #             y_scores[index] = y_ture
    #         else:
    #             y_scores[index] =  0   # y_trues[-1] - sim*0.1 # sim*0.1 is a penalty for this target's relavence
    #             if y_scores[index]<=0: 
    #                 y_scores[index] = 0

    #     y_trues = np.array(y_trues).reshape((1,-1))
    #     y_scores = np.array(y_scores).reshape((1,-1))

    #     count = 0 
    #     try:
    #         tmp_NDCG = ndcg_score(y_trues,y_scores)
    #         print(tmp_NDCG)
    #     except:
    #         print('hello world')
    #         tmp_NDCG = 1 # when  y_trues = [[1]] y_scores = [[1]] , it will run an error

    #     NDCG+=tmp_NDCG
    
    # NDCG = NDCG/size
    # return NDCG,count
    
  
        
        
    



def adp_neighbor_search(query_dict,ground_truth,vec_index,label_ix_dict,X,threshold=0.8):
    """
    Advanced Search neighbors for queryset according to cosine similarity among labels' embedding, when returned 
    target has a high levenshtein similarity with cue, disgard it. 
    example: lev_sim(give up, giveing up) > 0.8, disgard it

    Parameters
    ----------
    query_dict : dict
        A dictionary whose key is both in ground_truth and CSKG, value is the embedding 
        value according to CSKG
    
    ground_truth: dict
        A dictionary whose key is a cue's label, value is a list containing cue's similar targets 
        in decreasing order of similarity

    vec_index: faiss.swigfaiss_avx2.IndexFlat
        A IndexFlat that keeps the index for the label embeddings. The index number of each label is 
        determined according to the order of adding, for example, the first index is 0.
    
    label_ix_dict: dict
        A dictionary whose key is the index number, value is the label. This dictionary is aimed at recording each label's 
        order for future mapping.
    
    X: int
        Indicates how many times the data is extracted from CSKG according to the ground truth's label's targets number
        example: if we set X = 1  , and for a cue 'turtle' in USF_FAN_dict, there exists 18 similar targets for this cue,
        then we will search 18 similar labels for 'turtle' in CSKG  

    threshold: float [default:0.8]
        If the return target has a similarity equal or more than 0.8 with cue, then disgard this one

    Returns
    -------
    neighbor_dict:
        A dictionary whose key is a label in CSKG, value is a list containing the label's similar targets 
        in decreasing order of cosine similarity.
    """ 
    neighbor_dict = {}

    for query_label in tqdm(query_dict,total=len(query_dict),ncols=80):
        ### get the number of  target for this label/cue in ground turth
        target_num = len(ground_truth[query_label])
        count = 0
        not_satisfy = True
        while not_satisfy:
            neighbor_dict[query_label] = []
            query_mat = query_dict[query_label]
            # index is a row vector contains the index of similar labels
            cos_sim, index = vec_index.search(query_mat, target_num*(X+count)+1)  
            for ix in index[0][1:]:
                target = label_ix_dict[ix]
                if rltk.similarity.levenshtein.levenshtein_similarity(query_label,target) < threshold:
                    neighbor_dict[query_label].append(label_ix_dict[ix])

            if len(neighbor_dict[query_label]) >= X * target_num:
                not_satisfy = False
                neighbor_dict[query_label] = neighbor_dict[query_label][:target_num]
                break
            count+=1

    return neighbor_dict

if __name__ == "__main__":
    
    cue_target = 'input/cue-target.xml'
    cskg_connected = 'input/cskg_connected.tsv'
    bert_embs = 'input/bert-nli-large-embeddings.tsv.gz'
    kgtk_embs = 'input/trans_log_dot_0.1.tsv.gz' 

    print("load common dataset...")
    USF_FAN_dict = xml_load(cue_target)
    CSKG_label_dict,CSKG_inv_dict = cskg_load(cskg_connected)

    ### Graph Embedding
    graph_res = {}
    print("process graph embeddings on CSKG...")
    graph_embed_dict = graph_emb_load(kgtk_embs) 
    graph_label_emb = get_node_emb(graph_embed_dict,CSKG_label_dict)
    graph_index,graph_label_ix = build_index(graph_label_emb)
    graph_query_dict =  create_queryset(USF_FAN_dict,CSKG_label_dict,graph_label_emb)

    print("searching neighbors for CSKG labels...")
    Xs = [1,2,3,5,10]
    for X in Xs:
        neighbor_dict = neighbor_search(graph_query_dict,USF_FAN_dict,graph_index,graph_label_ix,X)
        hit = cal_hits(USF_FAN_dict,neighbor_dict)
        MAP = cal_map(USF_FAN_dict,neighbor_dict)
        MRR = cal_mrr(USF_FAN_dict,neighbor_dict)
        print(f"hit@{X}X: {hit}")
        print(f"MAP@{X}X: {MAP}")
        print(f"MRR@{X}X: {MRR}")
        graph_res[f"hit@{X}X"] = hit
        graph_res[f"MAP@{X}X"] = MAP
        graph_res[f"MRR@{X}X"] = MRR

    with open('graph_res.json','w') as f:
        json.dump(graph_res,f)

    ### Text Embedding
    text_res = {}
    print("process text embeddings on CSKG...")
    text_node_emb = txt_emb_load(bert_embs)
    text_label_emb = get_node_emb(text_node_emb,CSKG_label_dict)
    text_index,text_label_ix = build_index(text_label_emb)
    text_query_dict =  create_queryset(USF_FAN_dict,CSKG_label_dict,text_label_emb)

    print("searching neighbors for CSKG labels...")
    Xs = [1,2,3,5,10]
    for X in Xs:
        neighbor_dict = neighbor_search(text_query_dict,USF_FAN_dict,text_index,text_label_ix,X)
        hit = cal_hits(USF_FAN_dict,neighbor_dict)
        MAP = cal_map(USF_FAN_dict,neighbor_dict)
        MRR = cal_mrr(USF_FAN_dict,neighbor_dict)
        print(f"hit@{X}X: {hit}")
        print(f"MAP@{X}X: {MAP}")
        print(f"MRR@{X}X: {MRR}")
        text_res[f"hit@{X}X"] = hit
        text_res[f"MAP@{X}X"] = MAP
        text_res[f"MRR@{X}X"] = MRR

    with open('text_res.json','w') as f:
        json.dump(text_res,f)

    thresholds = [0,6,0.7,0.8,0.9]

    for threshold in thresholds:
        text_res2 = {}
        for X in Xs:
            neighbor_dict = adp_neighbor_search(text_query_dict,USF_FAN_dict,text_index,text_label_ix,X,threshold=threshold)
            hit = cal_hits(USF_FAN_dict,neighbor_dict)
            MAP = cal_map(USF_FAN_dict,neighbor_dict)
            MRR = cal_mrr(USF_FAN_dict,neighbor_dict)
            print(f"hit@{X}X: {hit}")
            print(f"MAP@{X}X: {MAP}")
            print(f"MRR@{X}X: {MRR}")
            text_res2[f"hit@{X}X"] = hit
            text_res2[f"MAP@{X}X"] = MAP
            text_res2[f"MRR@{X}X"] = MRR

        with open(f'text_res_{threshold}.json','w') as f:
            json.dump(text_res,f)

