from itertools import islice
import gzip
import os
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.metrics.cluster import adjusted_rand_score
from sentence_transformers import SentenceTransformer

"""
A set of util functions for CSKG relation analysis
"""

rel_template = {
    '/r/IsA': 'is a',
    '/r/SimilarTo': 'is similar to',
    '/r/Synonym': 'is same as',
    '/r/Antonym': 'is opposite to',
    '/r/RelatedTo': 'is related to',
    '/r/FormOf': 'is form of',
    '/r/AtLocation': 'is located at',
    '/r/DerivedFrom': 'is derived from',
    '/r/HasProperty': 'has property',
    '/r/DefinedAs': 'is defined as',
    '/r/EtymologicallyRelatedTo': 'is etymologically related to',
    '/r/InstanceOf': 'is a', # == '/r/IsA'
    '/r/dbpedia/genre': 'is a type of',
    '/r/CapableOf': 'is able to',
    '/r/PartOf': 'is a part of',
    '/r/MadeOf': 'is made of',
    '/r/ReceivesAction': 'can receive the action',
    '/r/HasA': 'has a',
    '/r/UsedFor': 'is used for',
    '/r/NotHasProperty': 'not has property',
    '/r/CausesDesire': 'causes desire to',
    '/r/dbpedia/occupation': 'occupation is', # career
    '/r/dbpedia/language': 'language is',
    '/r/HasSubevent': 'can have event',
    '/r/HasContext':'has context',
    '/r/LocatedNear':'is located near', ## .... lots of subpro .. shall I need to tag for each one?
    '/r/DistinctFrom': 'is different from',
    '/r/dbpedia/influencedBy': 'is influenced by',
    '/r/MannerOf': 'is manner of',
    'mw:MayHaveProperty': 'may have property', # similar to has property
    '/r/Entails': 'entails', # similat to cause
    '/r/dbpedia/field': 'belongs to field of',
    '/r/dbpedia/genus': 'belongs to genus of',
    '/r/HasPrerequisite': 'has prerequisite of',
    '/r/dbpedia/capital':"capital is",
    '/r/NotCapableOf': 'is not able to',
    '/r/dbpedia/product': 'is product of',
    '/r/MotivatedByGoal': 'is motivated by goal',
    '/r/Desires': 'desire to',
    '/r/Causes':'causes',
    '/r/HasFirstSubevent': 'starts with',
    '/r/HasLastSubevent': 'ends with',
    '/r/NotDesires': 'doest not desire to',
    '/r/dbpedia/knownFor':'is known for',
    '/r/CreatedBy': 'is created by',
    '/r/dbpedia/leader': 'has the leader',
    '/r/EtymologicallyDerivedFrom': 'is etymologically derived from',
    '/r/SymbolOf': 'is symbol of', # similar to 'is'
    'at:xAttr': 'has attribute',
    'at:xEffect': 'causes', # same as  '/r/Causes'
    'at:xIntent': 'wants to',
    'at:xReact':'feels',
    'at:xWant': 'wants to',
    'at:oReact': 'feels',
    'at:oWant': 'wants to',
    'at:xNeed' : 'needs',
    'at:oEffect': 'causes',
    'fn:HasFrameElement':'has frame element',
    'fn:HasLexicalUnit': 'has lexical unit',
    'fn:InheritsFrom': 'inherits from',
    'fn:IsInheritedBy' : 'is inherited by',
    'fn:IsUsedBy': 'is used by',
    'fn:PerspectiveOn': 'has perspective on',
    'fn:Uses': 'uses',
    'fn:IsPerspectivizedIn': 'is perspectivized in',
    'fn:Precedes':'precedes',
    'fn:SubframeOf': 'is subframe of',
    'fn:IsPrecededBy': 'is preceded by',
    'fn:ReframingMapping': 'reframing mapping',
    'fn:HasSubframe': 'has subframe',
    'fn:SeeAlso': 'see also',
    'fn:IsCausativeOf': 'causes', # same as  '/r/Causes'
    'fn:IsInchoativeOf': 'is inchoative of', # a little bit like 'cause'
    'fn:Metaphor': 'is similar to', # same as similar to
    'fn:HasSemType': 'has sem type', # same type???
    'fn:fe:ExcludesFE': 'excludes fe',# ?
    'fn:fe:RequiresFE': 'requires fe',
    'fn:st:RootType': "root type is",
    'fn:st:SuperType': "super type is",
    'fn:st:SubType': "sub type is",
}

def get_edge(input_file): 
    """
    Get edges' information from CSKG raw data.

    Parameters
    ----------
    input_file : str
        The file path of cskg_connected.tsv
        each line's format is: id  node1 relation  node2  node1;label  node2;label  relation;label  
    
    Returns
    -------
    edge_list: list
        A list contain multiple tuples kepping each edge's nodes and relation information 
        each tuple's format is  (edge_id, node1_lbl, rel_lbl, node2_lbl, rel_meta)
    """ 
    edge_list = []
    with open(input_file) as f:
        for line in islice(f, 1, None): # ignore header
            content = line.split('\t')
            edge_id =  content[0]  # edge id
            node1_lbl = content[4] # node1 label
            node2_lbl = content[5] # node2 label
            rel_meta = content[2] # relation id
            rel_lbl = content[6]   # relation label

            # label may have multiple times, here just use the first one
            node1_lbl = node1_lbl.split('|')[0]
            node2_lbl = node2_lbl.split('|')[0]
            rel_lbl = rel_lbl.split('|')[0]
            # if one of them is empty, skip this edge
            if node1_lbl == '' or node2_lbl == '' or rel_lbl == '':
                continue   
            edge_list.append((edge_id,node1_lbl,rel_lbl,node2_lbl,rel_meta)) 
    return edge_list 

def rel_mapping(edge_list):
    """
    Get the relationship between relation ID and relation labels
    one reltaion ID may have multiple relation label
    example: '/r/IsA' has labels like 'is a', 'subclass of', 'instance of' ...

    Parameters
    ----------
    edge_list : list 
        A list contain multiple tuples kepping each edge's nodes and relation information 
        each tuple's format is  (edge_id, node1_lbl, rel_lbl, node2_lbl, rel_meta)  
    
    Returns
    -------
    rel_dict: dict
        A dictionary whose key the relation ID , value keeps the relation label accoring to the relation ID.
        The value is also a dictionary whose key is the relation label, the value is the occurrence such relation
        label appears on CSKG
        example: '/r/IsA': {'is a': 242358, 'subproperty of': 1, 'subclass of': 47501, 'instance of': 26685} 
    """ 
    rel_dict = {}
    lbl_dict = {}
    for edge in edge_list:
        rel_meta = edge[-1] 
        rel_lbl = edge[2] 
        lbl_dict[rel_lbl] = lbl_dict.get(rel_lbl,0)+1
        rel_dict[rel_meta] = rel_dict.get(rel_meta,{})
        rel_dict[rel_meta][rel_lbl] = lbl_dict[rel_lbl] 
    return rel_dict

def create_lexi(edge_list,rel_template,output_file):
    """
    Create lexicalization on CSKG's edges according to rel_template and its nodes
    example: if a edge's relation is 'is a' , then we can lexicalize this edge to a sentence like
    'node1_label is a node2_label'

    Parameters
    ----------
    edge_list : list 
        A list contain multiple tuples kepping each edge's nodes and relation information 
        each tuple's format is  (edge_id, node1_lbl, rel_lbl, node2_lbl, rel_meta)  

    rel_template: dict
        A dictionary made manually keeps the template for different relation types
    
    output_file: str:
        The file path for output of lexicalization on  CSKG's edges
        each line's format is: edge_id   lexicalization  sentence 
        example: 
        '/c/en/0.22_inch_calibre-/r/IsA-/c/en/5.6_millimetres-0000'  'is a'   '0.22 inch calibre is a 5.6 millimetres'
        
    Returns
    -------
    edge_sent_list: list
        A list contain multiple tuples kepping each edge's id, lexicalization and generated sentence
        each tuple's format is  (edge_id, lexicalization, sentence)       
    """ 
    edge_sent_list = []
    with open(output_file,'w') as f:  
        # format for cskg_info : edge_id, node1_label, relation_label, node2_label, , relation_meta
        for edge in edge_list:
            edge_id = edge[0] 
            node1_label = edge[1]
            node2_label = edge[3]
            rel_meta = edge[4] 
            lexicalization = rel_template[rel_meta]
            sentence =  node1_label + ' ' + lexicalization + ' '  + node2_label
            f.write(f"{edge_id}\t")  #  f.write(f"{{{edge_id}}}\t")
            f.write(f"{lexicalization}\t")
            f.write(f"{sentence}\n")  # f.write(f"{{{sentence}}}\n")
            edge_sent_list.append((edge_id,lexicalization,sentence))

    return edge_sent_list

def get_sent_emb(model_name,edge_sent_list,output_file):
    """
    Proceed sentence embedding for edges by using SentenceTransformer

    Parameters
    ----------
    model_name : str
        Which model should be used
    
    edge_sent_list: list
        A list contains multiple tuples kepping each edge's id, lexicalization and generated sentence
        each tuple's format is  (edge_id, lexicalization, sentence) 

    output_file: str
        The file path for output of sentence embedding on  CSKG's edges
        each line's format is: edge_id   tag  sent_embedding
        example: 
        '/c/en/0.22_inch_calibre-/r/IsA-/c/en/5.6_millimetres-0000'  'edge-embedding'  '0.1,0.2,0.3...'
    
    Returns
    -------
    edge_embeds: list
        A list contains multiple tupels, each tuple contians the edge's id and edge's sentence embedding
        example: [ (edge1, value), (edge2,value2)] 
    """ 

    model = SentenceTransformer(model_name)
    sentences = [i[-1] for i in edge_sent_list]
    sent_embs = model.encode(sentences)

    edge_embeds = []
    fout = open(output_file,'w')   
    tag = 'edge-embedding'
    for edge, embedding in zip(edge_sent_list, sent_embs):
        edge_id =  edge[0]
        value = [str(float(i)) for i in embedding]
        value = ','.join(value)
        fout.write(edge_id+'\t')
        fout.write(tag+'\t')
        fout.write(value+'\n') 

        edge_embeds.append((edge_id,embedding))
    fout.close()

    return edge_embeds

def edge_cluster(edge_embeds,cluster_num=13):
    """
    Cluster edges on CSKG according to the sentence embeddings

    Parameters
    ----------
    edge_embeds: list   
        A list contains multiple tupels, each tuple contians the edge's id and edge's sentence embedding
        example: [ (edge1, value), (edge2,value2)] 
    
    Returns
    -------
    clstr_res_method: dict
        A Dictionary whose key is the egde id, the value is the predicted cluster label  by k-means  
    """  
    edge_list = [edge[0] for edge in edge_embeds]  # keep each edge's id
    X_train = [edge[1] for edge in edge_embeds]    # keep each edge's embeddings
    prediction = []
    model = KMeans(n_clusters=cluster_num,init='k-means++',verbose=0)
    model.fit(X_train)
    pred_labels = model.labels_

    clstr_res_method = {}
    for index,edge_id in enumerate(edge_list):
        clstr_res_method[edge_id] = pred_labels[index]
    # clstr_res_method = list(zip(edge_list,pred_labels))

    return clstr_res_method

def load_clstr_hand(input_file):
    """
    Load the cluster result on CSKG edges by their relations (manually).
    
    Parameters
    ----------
    input_file : str 
        The file path of cskg_connected_dim.tsv.gz
        each line's format is: 
        id	node1 relation node2 node1;label node2;label relation;label	relation;dimension source sentence 

    Returns
    -------
    clstr_res_hand: dict
        A Dictionary whose key is the egde id, the value is the cluster label    
    """    
    clstr_res_hand = {}
    with gzip.open(input_file,'rt') as f:
        for line in islice(f, 1, None): # ignore header  # 5822389 lines
            content = line.split('\t')
            edge_id = content[0]
            cluster_label = content[7]
            clstr_res_hand[edge_id] = cluster_label
           
    return clstr_res_hand

def adj_rank_index(clstr_res_method,clstr_res_hand):
    """
    Calculate  adjusted rand index metric between clusters' label computed automatically by the embeddings and clusters'
    label recognized by huam
    
    Parameters
    ----------
    clstr_res_method: dict
        A Dictionary whose key is the egde id, the value is the predicted cluster label  by k-means  

    clstr_res_hand: dict
        A Dictionary whose key is the egde id, the value is the cluster label by human recognition

    Returns
    -------
    adj_rank_score: float
        Similarity score between -1.0 and 1.0. Random labelings have an adj_rank_score 
        close to 0.0. 1.0 stands for perfect match.
    """    

    X = []
    Y = []
    for edge_id,cluster_label in clstr_res_method.items():
        if edge_id in clstr_res_hand:
            X.append(cluster_label)
            Y.append(clstr_res_hand[edge_id])
        else:
            continue

    adj_rank_score = adjusted_rand_score(X,Y)
    return adj_rank_score

def visualisation(clstr_res_method, edge_embeds, log_path):
    """
    Generate a log foloder for tensorboard to display visualization for edge embeddings
    
    Parameters
    ----------
    clstr_res_method: dict
        A Dictionary whose key is the egde id, the value is the predicted cluster label  by k-means  

    edge_embeds: list
        A list contains multiple tupels, each tuple contians the edge's id and edge's sentence embedding
        example: [ (edge1, value), (edge2,value2)] 

    log_path: str
        folder path for storing these information

    Returns
    -------
    None
    """    
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    edge_ids,embeddings = [],[]

    for index,edge in enumerate(edge_embeds):
        edge_id = edge[0]
        edge_emb = edge[1]
        pred_label = clstr_res_method[edge_id]
        embeddings.append(edge_emb) 
        edge_ids.append((edge_id,pred_label))
    embeddings = np.array(embeddings)

    with tf.Session() as sess:
        # assign embeddings to tf var
        x = tf.Variable([0.0], name='edge_embedding_roberta')
        place = tf.placeholder(tf.float32, shape=[len(edge_ids), len(embeddings[0])])
        set_x = tf.assign(x, place, validate_shape=False)
        sess.run(tf.global_variables_initializer())
        sess.run(set_x, feed_dict={place: embeddings})

        # store metedata to log path
        with open(log_path + '/metadata.tsv', 'w') as f:
            f.write('Edge ID\tCluster\n')
            for edge in edge_ids:
                edge_id = edge[0]
                edge_clusetr = edge[1]
                f.write(str(edge_id) + '\t' + str(edge_clusetr)+'\n' )

        # summary writing
        summary_writer = tf.summary.FileWriter(log_path)   
        # projector configuration
        config = projector.ProjectorConfig()
        emb_conf = config.embeddings.add()
        emb_conf.tensor_name = x.name
        emb_conf.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(summary_writer, config)
        # save model.ckpt
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(log_path, 'model.ckpt'), 1)




if __name__ == "__main__":

    cskg_connected = 'input/cskg_connected.tsv'
    cskg_lexicalized = 'output/cskg_lexicalized.tsv'
    edge_embeddings = 'output/edge_embeddings_bert.tsv'
    edge_embeddings2 = 'output/edge_embeddings_robert.tsv'
    cskg_connected_dim = 'input/cskg_connected_dim.tsv.gz'
    log_path = 'output/log'


    # edge_list = get_edge(cskg_connected)

    # rel_dict = rel_mapping(edge_list) 

    # edge_sen_list = create_lexi(edge_list,rel_template,cskg_lexicalized)

    # edge_embeds1 = get_sent_emb('bert-large-nli-stsb-mean-tokens',edge_sen_list,edge_embeddings)
    # edge_embeds2 = get_sent_emb('roberta-large-nli-stsb-mean-tokens',edge_sen_list,edge_embeddings2)
   
