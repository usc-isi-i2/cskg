import gzip
import os 
from tqdm import tqdm
import numpy as np
from itertools import islice
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

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

def load_sent_emb(input_file):
    """
    Load  existing sentence embedding for edges

    Parameters
    ----------
    input_file: str
        The file path  of sentence embedding on  CSKG's edges
        each line's format is: edge_id   tag  sent_embedding
        example: 
        '/c/en/0.22_inch_calibre-/r/IsA-/c/en/5.6_millimetres-0000'  'edge-embedding'  '0.1,0.2,0.3...'

    Returns
    -------
    edge_embeds: list
        A list contains multiple tupels, each tuple contians the edge's id and edge's sentence embedding
        example: [ (edge1, value), (edge2,value2)] 
    """ 
    edge_embeds = []
    with tqdm(total=os.path.getsize(input_file)) as pbar:
        with open(input_file) as f:
            for line in f: # don't use f.readlines() , file is too large , spend too much read it once 
                content = line.split('\t')
                edge_id = content[0] 
                edge_value = content[-1]
                edge_embedding = [float(i) for i in edge_value.split(',')]
                edge_embeds.append((edge_id,edge_embedding))
                pbar.update(len(line))

    return edge_embeds

def edge_cluster(edge_embeds,output_file,cluster_num=13):
    """
    Cluster edges on CSKG according to the sentence embeddings

    Parameters
    ----------
    edge_embeds: list   
        A list contains multiple tupels, each tuple contians the edge's id and edge's sentence embedding
        example: [ (edge1, value), (edge2,value2)] 

    output_file: str   
        tsv file path for stroing cluster results, each line contians two items edge id and 
        predicted separated by tab
    
    Returns
    -------
    clstr_res: dict
        A Dictionary whose key is the egde id, the value is the predicted cluster label  by k-means  
    """  
    edge_list = [edge[0] for edge in edge_embeds]  # keep each edge's id
    X_train = [edge[1] for edge in edge_embeds]    # keep each edge's embeddings
    prediction = []
    model = KMeans(n_clusters=cluster_num,init='k-means++',verbose=0)
    model.fit(X_train)
    pred_labels = model.labels_

    clstr_res = {}
    fw = open(output_file,'w')

    for index,edge_id in enumerate(edge_list):
        cls_label = pred_labels[index]
        clstr_res[edge_id] = cls_label
        fw.write(f"{edge_id}\t{cls_label}\n")

    fw.close()

    return clstr_res

def load_clstr_auto(input_file):
    """
    Load the cluster result on CSKG edges by k means
    
    Parameters
    ----------
    input_file : str 
        The file path of cskg_connected_dim.tsv.gz
        each line's format is: id label

    Returns
    -------
    cluster_res: dict
        A Dictionary whose key is the egde id, the value is the cluster label    
    """    
    cluster_res = {}
    with open(input_file) as f:
        for line in f.readlines(): 
            content = line.split('\t')
            edge_id = content[0]
            cluster_label = content[1].replace('\n','')
            cluster_res[edge_id] = cluster_label

    return cluster_res

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
    manually_res: dict
        A Dictionary whose key is the egde id, the value is the cluster label    
    """    
    manually_res = {}
    with gzip.open(input_file,'rt') as f:
        for line in islice(f, 1, None): # ignore header  # 5822389 lines
            content = line.split('\t')
            edge_id = content[0]
            cluster_label = content[7]
            manually_res[edge_id] = cluster_label
           
    return manually_res

def adj_rank_index(clstr_res_auto,clstr_res_hand):
    """
    Calculate  adjusted rand index metric between clusters' label computed automatically by the embeddings and clusters'
    label recognized by huam
    
    Parameters
    ----------
    clstr_res_auto: dict
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
    for edge_id,cluster_label in clstr_res_auto.items():
        if edge_id in clstr_res_hand:
            X.append(cluster_label)
            Y.append(clstr_res_hand[edge_id])
        else:
            continue

    adj_rank_score = adjusted_rand_score(X,Y)
    return adj_rank_score

def visualisation(clstr_res, edge_embeds, log_path,board_name):
    """
    Generate a log foloder for tensorboard to display visualization for edge embeddings
    
    Parameters
    ----------
    clstr_res: dict
        A Dictionary whose key is the egde id, the value is the predicted cluster label  either by k-means or human

    edge_embeds: list
        A list contains multiple tupels, each tuple contians the edge's id and edge's sentence embedding
        example: [ (edge1, value), (edge2,value2)] 

    log_path: str
        folder path for storing these information
    
    board_name: str
        name of tensorboard projector

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
        pred_label = clstr_res.get(edge_id,None)
        if pred_label is None:
            continue
        embeddings.append(edge_emb) 
        edge_ids.append((edge_id,pred_label))
    embeddings = np.array(embeddings)

    with tf.Session() as sess:
        # assign embeddings to tf var
        x = tf.Variable([0.0], name=board_name)
        place = tf.placeholder(tf.float32, shape=[len(edge_ids), len(embeddings[0])])
        set_x = tf.assign(x, place, validate_shape=False)
        sess.run(tf.global_variables_initializer())
        sess.run(set_x, feed_dict={place: embeddings})

        # store metedata to log path
        with open(log_path + '/metadata.tsv', 'w') as f:
            f.write('Edge ID\tCluster\n')
            for edge in edge_ids:
                edge_id = edge[0]
                edge_clusetr = edge[1].replace('\n','')
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

def tsne(edge_embeds,perplexity=30,learning_rate=200,metric='euclidean',init='random'):
    """
    Calculate  adjusted rand index metric between clusters' label computed automatically by the embeddings and clusters'
    label recognized by huam
    
    Parameters
    ----------
    edge_embeds: list
        A list contains multiple tupels, each tuple contians the edge's id and edge's sentence embedding
        example: [ (edge1, value), (edge2,value2)] 

    perplexity: float, optional (default: 30)
        The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. 
        Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. Different 
        values can result in significanlty different results.

    learning_rate: float, optional (default: 200.0)
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If the learning rate is too high, the data 
        may look like a ‘ball’ with any point approximately equidistant from its nearest neighbours. If the learning rate 
        is too low, most points may look compressed in a dense cloud with few outliers. If the cost function 
        gets stuck in a bad local minimum increasing the learning rate may help.

    metric: str or callable, optional
        The metric to use when calculating distance between instances in a feature array. If metric is a string, 
        it must be one of the options allowed by scipy.spatial.distance.pdist for its metric parameter, 
        or a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is “precomputed”, X is assumed to be a 
        distance matrix. Alternatively, if metric is a callable function, it is called on each pair of instances (rows) 
        and the resulting value recorded. The callable should take two arrays from X as input and return a value indicating
        the distance between them.  The default is “euclidean” which is interpreted as squared euclidean distance.

    init: str or numpy array, optional (default: “random”)
        Initialization of embedding. Possible options are ‘random’, ‘pca’, and a numpy array of shape 
        (n_samples, n_components). PCA initialization cannot be used with precomputed distances and is 
        usually more globally stable than random initialization.

    Returns
    -------
    embedding_ids: list
        A list contains edges id 

    new_embed: array-like,shape (n_samples, n_components)
        Stores the embedding vectors， each vector's order is the same as embedding_ids

    kl_divergence: float
        Kullback-Leibler divergence after optimization.
    """    
    embedding_ids = [edge_info[0] for edge_info in edge_embeds]
    embedding_mat = [edge_info[1] for edge_info in edge_embeds]
    embedding_mat = np.array(embedding_mat)

    tsne = TSNE(n_components=2,perplexity=perplexity, 
    learning_rate=learning_rate,metric=metric,init=init,n_jobs=-1) # use all cores
    tsne.fit_transform(embedding_mat)
    new_embed = tsne.embedding_
    kl_divergence = tsne.kl_divergence_

    return embedding_ids,new_embed,kl_divergence


