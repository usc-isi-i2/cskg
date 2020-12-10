import os
import json 
from itertools import islice
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


########    ########    ########    ########    ########    ########    ############
# Wrapper for 'CSKG Edge Analysis.ipynb'
# This scrpit has the same code with 'CSKG Edge Analysis.ipynb'
########    ########    ########    ########    ########    ########    ############

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


############################ Load data ##############################################
### load cskg label and relation
def cskg_lexicalize(input_file): # cskg_connected.tsv   
    cskg_info = []
    with open(input_file) as f:
        for line in islice(f, 1, None): # ignore first line 
            content = line.split('\t')
            id_=  content[0]
            res_meta = content[2]
            node1_lbl = content[4]
            node2_lbl = content[5]
            res_lbl = content[6]
            
            reterive = lambda x: x.split('|')[0]
            node1_lbl = reterive(node1_lbl)
            node2_lbl = reterive(node2_lbl)
            res_lbl = reterive(res_lbl)
            
            if node1_lbl == '' or node2_lbl == '' or res_lbl== '':
                # if one of them is empty, then skip the edge
                continue                
            cskg_info.append((id_,node1_lbl,res_lbl,node2_lbl,res_meta))         
    return cskg_info 

# store res metadata and label infomation
def rel_info(cskg_info):
    rel_meta_dict = {}
    rel_lbl_dict = {}
    
    for line in cskg_info:
        rel_meta = line[-1]  # example: '/r/Is a'
        rel_label = line[2]  # example: 'is a '
        rel_lbl_dict[rel_label] = rel_lbl_dict.get(rel_label,0)+1
        rel_meta_dict[rel_meta] = rel_meta_dict.get(rel_meta,{})
        rel_meta_dict[rel_meta][rel_label] = rel_lbl_dict[rel_label]      
    return rel_meta_dict



def create_cskg_lexi(cskg_info,rel_template,output_file):
    edge_info = []
    with open(output_file,'w') as f:  
        # format for cskg_info : edge_id, node1_label, relation_label, node2_label, , relation_meta
        for edge in cskg_info:
            edge_id = edge[0] 
            node1_label = edge[1]
            node2_label = edge[3]
            res_meta = edge[4] 
            lexicalization = rel_template[res_meta]
            sentence =  node1_label + ' ' + lexicalization + ' '  + node2_label
            
            f.write(f"{edge_id}\t")  #  f.write(f"{{{edge_id}}}\t")
            f.write(f"{lexicalization}\t")
            f.write(f"{sentence}\n")  # f.write(f"{{{sentence}}}\n")
            edge_info.append((edge_id,lexicalization,sentence))
    return edge_info

# output embedding result
def gen_edge_embed(edge_info, sentence_embeddings,output_file):
    # format edge_id   edge-embedding   embedding
    edge_embedding = []
    fout = open(output_file,'w')   
    tag = 'edge-embedding'
    for edge, embedding in zip(edge_info, sentence_embeddings):
        id_ = edge[0]
        value = [str(float(i)) for i in embedding]
        value = ','.join(value)
        fout.write(id_+'\t')
        fout.write(tag+'\t')
        fout.write(value+'\n') 
        edge_embedding.append((id_,embedding))
        
    fout.close()
    return edge_embedding


def visualisation(edge_ids, embeddings, log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    with tf.Session() as sess:
        # assign embeddings to tf var
        x = tf.Variable([0.0], name='edge embedding')
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
    edge_embeddings = 'output/edge_embeddings.tsv'
    log_path = 'output/log'

    #1. load cskg data 
    cskg_info = cskg_lexicalize(cskg_connected)

    #2. create a relation meta dictionary , its key is the relation node and values is the dict
    #  of relation labels with frequency
    # example:
    # rel_meta_dict['/r/IsA']: {'is a': 242358, 'subproperty of': 1, 'subclass of': 47501, 'instance of': 26685}
    rel_meta_dict = rel_info(cskg_info)

    #3. represent each of the edges in TSV format: edge_id   lexicalization   sentence   
    edge_info = create_cskg_lexi(cskg_info,rel_template,cskg_lexicalized)

    #4. use the pre-trained model bert-base-nli-mean-tokens and do sentence emebdding
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    entence_embeddings = model.encode(sentences) # may take 1h

    #5. output embedding result
    edge_embedding = gen_edge_embed(edge_info, sentence_embeddings,edge_embeddings)


    #6.Use K-means to do clusetring
    edge_list = [edge[0] for edge in edge_embedding]  # keep each edge's id
    X_train = [edge[1] for edge in edge_embedding]    # keep each edge's embeddings   

    model = KMeans(n_clusters=13,init='k-means++',verbose=0)
    model.fit(X_train)
    pred_labels = model.labels_  
    

    #7. use tensorflow projector to display visualization
    embeddings = []
    edge_ids = []
    for inde,edge in enumerate(edge_embedding):
        pred_label = pred_labels[inde]
        embeddings.append(edge[1]) 
        edge_ids.append((edge[0],pred_label))
        
    embeddings = np.array(embeddings)
    visualisation(edge_ids, embeddings, log_path)
