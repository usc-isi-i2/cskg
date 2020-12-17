from sentence_transformers import SentenceTransformer
import rltk, re, random, heapq, csv
from nltk.corpus import wordnet as wn
from itertools import combinations, cycle

################################################################

"""
load or write data
sample file is kgtk_conceptnet.tsv or wn_gold_all.tsv
"""
def load_file(filename, encoding="UTF-8"):
    """
    load data
    token is split by "\t"
    return head, lines
    """
    with open(filename, "r",encoding=encoding) as f:

        head = f.readline().strip().split("\t")
        lines = []

        for line in f:
            lines.append(line.strip().split("\t"))
    return head, lines

def synset2str(syn):
    """
    transfer synset to str in gold file
    input: synset or string
    output: wn:XX.X.X
    """
    if type(syn) == str:
        return "wn:"
    return "wn:"+ syn.name()

# write data to file
def write_prediction(filename, lines,encoding="UTF-8"):
    """
    write prediction to file
    new_head = ['node1;label','relation','node2;label','node1','node2']
    input: 
        filename: The path to store data
        lines: The data to write 
        (data structure of each line (node1label, relation, node2label, node1_synset, node2_synset))
    """
    with open(filename, "a", newline='',encoding=encoding) as f:
        writer = csv.writer(f, delimiter='\t')
        #writer.writerow(new_head)
        for line in lines:
            writer.writerow([line[0],line[1],line[2],synset2str(line[3]),synset2str(line[4])])

def write_gold(filename, lines, encoding="UTF-8"):
    """
    write golden data to file
    new_head = ['node1;label','relation','node2;label','node1','node2']
    input: 
        filename: The path to store data
        lines: The data to write 
        (data structure of each line (node1label, relation, node2label, node1id, node2id))
    """
    new_head = ['node1;label','relation','node2;label','node1','node2']
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(new_head)
        for line in lines:
            writer.writerow([line[0],line[1],line[2],line[3],line[4]])
################################################################

################################################################
# statistc for result

def no_synset_count(cn_predict_1k):
    # check the count of no synset in result
    # count 1: no synset for label
    # count 2: no synset for record
    count1 = 0
    count2 = 0
    for record in cn_predict_1k:
        judge = [synset2str(record[3]) == "wn:",synset2str(record[4]) == "wn:"]
        
        if judge[0]:
            count1 += 1
            
        if judge[1]:
            count1 += 1
            
        if any(judge):
            count2 += 1
            
    return count1, count2
################################################################

################################################################
# Generate file used for node resolution
# column: 'node1;label','relation','node2;label','node1','node2'
def multiple_labels(node_labels,node_noid):
    """
    Obtain label whose leven distance is most close to the node id
    Based on levenshtein distance
    """

    node_res=[float("inf"),""]

    split_labels = node_labels.split("|")
    if len(split_labels) == 1:
        # if only one label exits, return this label
        return split_labels[0].replace('"',"").replace("\\'","'")


    for label in split_labels:
        # remove ""
        label = label.replace('"',"").replace("\\'","'")
        dis = rltk.levenshtein_distance(node_noid, label)
        temp = [dis,label]
        
        if temp < node_res:
            node_res  = temp
            
    return node_res[1]

def generate_gold_file(lines):
    # Transfer line to head 'node1;label','relation','node2;label','node1','node2'
    wn_gold_all = []
    i = 0
    for line in lines:
        #change column to node1;label, relation, node2;label, node1, node2
        node1_id = line[0]
        relation = line[1]
        node2_id = line[2]
        node1_labels = line[3]
        node2_labels = line[4]

        # modeify the node labels, check with leve distance

        node1_label = multiple_labels(node1_labels,node1_id)
        node2_label = multiple_labels(node2_labels,node2_id)

        wn_gold_all.append([node1_label, relation, node2_label, node1_id, node2_id])
        i += 1
        if i%10000==1:
            print(f"\r {i}/{len(lines)}", end="")
        
    return wn_gold_all

################################################################


################################################################
# Get Synsets
# use label name to obtain avaliable node id from WordNet NLTK
# example:
#  generate_candidates("happy") --->[Synset('happy.a.01'),Synset('felicitous.s.02'),Synset('glad.s.02'),Synset('happy.s.04')]
def place_ones(size, count):
    for positions in combinations(range(size), count):
        p = [0] * size

        for i in positions:
            p[i] = 1

        yield p
        
# permutations of list n without repitation
def permu(n):
    comb = []
    for i in range(n+1):
        comb += place_ones(n,i)
        
    return comb

def replace_str(string, replace_w, idx):
    # replace whitespace to _ or -
    return string[:idx] + replace_w +string[idx+1:]

def transfer_words(label):
    # accoording to the labels, generate label that can be recognized by wn interface from NLTK
    idx_list = [x for x, v in enumerate(label) if v == ' ']
    
    if not idx_list:
        #no whitespace in words
        yield label
    else:
        # whitespace in words
        combs = permu(len(idx_list))
        for comb in combs:
            for idx, status, in zip(idx_list, comb):
                if status:
                    label = replace_str(label, "-", idx)
                else:
                    label = replace_str(label, "_", idx)
            yield label
            
def generate_synsets(labels):
    # According to the generation of labels, obtain the synsets
    for label in labels:
        synsets = list(wn.synsets(label))
        
        if synsets:
            return synsets, label
        
    return [], label

def generate_candidates(label):
    candidates,_ = generate_synsets(transfer_words(label))
    return candidates
################################################################


################################################################
# define transfermation from predicate to sentence connection
# based on the file kgtk_conceptnet.tsv

word2sentence = {'/r/Antonym':"is antonym for", 
                 '/r/AtLocation': "is located at",
                 '/r/CapableOf':"is capable of",
                '/r/Causes':"causes",
                '/r/CausesDesire':"causes the desire of",
                '/r/CreatedBy':"is created by",
                '/r/DefinedAs': " is defined as",
                '/r/DerivedFrom': "is derived from",
                '/r/Desires':"desires",
                '/r/DistinctFrom':"is distinct from",
                "/r/Entails":"entails",
                '/r/EtymologicallyDerivedFrom':"is etymologically derived from",
                '/r/EtymologicallyRelatedTo': "is etymologically related to",
                '/r/FormOf':"is form of",
                '/r/HasA': "has a",
                '/r/HasContext': "has the context of",
                '/r/HasFirstSubevent': "has first subevent, ",
                '/r/HasLastSubevent':"has last subevent, ",
                '/r/HasPrerequisite': "has prerequisite, ",
                '/r/HasProperty': "has property, ",
                '/r/HasSubevent': "has subevent, ",
                '/r/InstanceOf': " is an instance of",
                '/r/IsA': "is a",
                '/r/LocatedNear': "is located nearby",
                '/r/MadeOf': "is made of",
                '/r/MannerOf':"has a manner of",
                '/r/MotivatedByGoal': "is motivated by goal",
                '/r/NotCapableOf': "is not capable of",
                '/r/NotDesires':"does not desire",
                '/r/NotHasProperty':"does not have property, ",
                '/r/PartOf': "is part of",
                '/r/ReceivesAction':"receives the action, ",
                '/r/RelatedTo':"is related to",
                '/r/SimilarTo':"is similar to",
                '/r/SymbolOf':"is a symbol of",
                '/r/Synonym':"is synonym for",
                '/r/UsedFor':"is used for",
                '/r/dbpedia/capital': "is the capital of",
                '/r/dbpedia/field':" is the field of",
                '/r/dbpedia/genre':"has genre,",
                '/r/dbpedia/genus':"has genus, ",
                '/r/dbpedia/influencedBy':"is influenced by",
                '/r/dbpedia/knownFor': "is known for",
                '/r/dbpedia/language':"is the language ",
                '/r/dbpedia/leader':"has the leader, ",
                '/r/dbpedia/occupation':"has the occupation, ",
                '/r/dbpedia/product':"has the product, "}

def line_sentence(line, word2sentence):
    # combine "label1 name" + "predicate sentence" + "label2 name"
    if len(line)>=6 and line[5] != "":
        return line[5] 
    sentence = line[0]+" "+word2sentence[line[1]]+" "+line[2]
    return sentence
################################################################


################################################################
# generate all label node id defination embeddings dict from file

def label2sentence2sent(label, model,label_synsets,sents_combine):
    temp = []

    candidates = generate_candidates(label)
    candits_sent = [ _.definition() for _ in candidates]
    
    sents_combine += candits_sent
    for candit in candidates:
        label_synsets.append([label,candit])
            
    return label_synsets,sents_combine

def candidates_embeddings(wn_gold, model):
    # generate label node id defination embeddings from file
    # output:{"label_name":[[node_id, embedding of node_id defination],[X,X],[X,X]]}
    
    # store label, synset
    label_synsets = []
    # store the defination sentence of synset
    sents_combine = []
    
    embeddings = dict()

    for line in wn_gold:
        label1 = line[0]
        label2 = line[2]
        
        label_synsets,sents_combine = label2sentence2sent(label1, model,label_synsets,sents_combine)
        label_synsets,sents_combine = label2sentence2sent(label2, model,label_synsets,sents_combine)

    # generate embedding of sentence
    sents_embed = model.encode(sents_combine)
    for label_synset, embed in zip(label_synsets,sents_embed):
        label, synset = label_synset
        
        temp = embeddings.get(label,[])
        temp.append([synset,embed])
        
        # generate embedding of label synset
        embeddings[label]=temp
    return embeddings
################################################################


################################################################
# define transfermation from predicate to limitation
# POS type for WordNet: ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
# same: node_id1 and node_id2 have same pos
# different: node_id1 and node_id2 have different pos
# label{X}_{pos_i}: nodeid x must be the pos, pos_i
word2limit = {'/r/Antonym':["same"], 
                 '/r/AtLocation': ["same","label1_n","label2_n"],
                 '/r/CapableOf':["label1_n"], #example sentence: "something can do somthing" or "somthing have somethingd"
                '/r/Causes':[],
                '/r/CausesDesire':[],
                '/r/CreatedBy':["label1_n"],
                '/r/DefinedAs': ["same"],
                '/r/DerivedFrom': ["same"],
                '/r/Desires':[],
                '/r/DistinctFrom':["same"],
                "/r/Entails":[],
                '/r/EtymologicallyDerivedFrom':["same"],
                '/r/EtymologicallyRelatedTo': ["same"],
                '/r/FormOf':["same"],
                '/r/HasA': ["same"],
                '/r/HasContext': ["label2_n"],
                '/r/HasFirstSubevent': ["same"],
                '/r/HasLastSubevent':["same"],
                '/r/HasPrerequisite': [],
                '/r/HasProperty': [],
                '/r/HasSubevent': ["same"],
                '/r/InstanceOf': ["same","label1_n","label2_n"],
                '/r/IsA': ["same"],
                '/r/LocatedNear':["same"],
                '/r/MadeOf': ["same", "label1_n","label2_n"],
                '/r/MannerOf':[],
                '/r/MotivatedByGoal': [],
                '/r/NotCapableOf': ["label1_n"],
                '/r/NotDesires':[],
                '/r/NotHasProperty':[],
                '/r/PartOf':["same"],
                '/r/ReceivesAction':["label2_v"],
                '/r/RelatedTo':["same"],
                '/r/SimilarTo':["same"],
                '/r/SymbolOf':[],
                '/r/Synonym':["same"],
                '/r/UsedFor':[],
                '/r/dbpedia/capital':["label1_n","label2_n"],
                '/r/dbpedia/field':["same"],
                '/r/dbpedia/genre':["same"],
                '/r/dbpedia/genus':["same"],
                '/r/dbpedia/influencedBy':[],
                '/r/dbpedia/knownFor': [],
                '/r/dbpedia/language':["label2_n"],
                '/r/dbpedia/leader':["same","label1_n","label2_n"],
                '/r/dbpedia/occupation':[],
                '/r/dbpedia/product':["same"]}

#Functions to check whether node id satisfy the predicate limitation
def nodeids_check(node_id1,node_id2,limit):
    
    if limit == "same":
        #node_id1 and node_id2 should have the same pos

        if not node_id1 or not node_id2:
            return True
        
        return node_id1.pos() == node_id2.pos()
    
    elif limit == "different":
        # different: node_id1 and node_id2 have different pos
        
        if not node_id1 or not node_id2:
            return True
        
        return node_id1.pos() != node_id2.pos()
        
    elif "_" in limit:
        # label{X}_{pos_i}: nodeid x must be the pos, pos_i
        
        label_tag, pos_tag = limit.split("_")
        
        if label_tag == "label1":
            if not node_id1:
                return True
            
            return node_id1.pos() == pos_tag
        
        elif label_tag == "label2":
            if not node_id2:
                return True
            
            return node_id2.pos() == pos_tag

def predicate_limitation_check(node_id1, node_id2, predicate, word2limit):
    # check whether node_id1 and node_id2 satisfy the limitation of predicate (e.g. node_id1 and node_id2 should have same pos)
    
    # Kinds of limitation predicate has
    limitations = word2limit[predicate]
    
    check_res = []
    for limit in limitations:
        temp = nodeids_check(node_id1, node_id2, limit)
        
        check_res.append(temp)
    
    # only if all requirements are satisfied, node_id1 and node_id2 can be used
    return all(check_res)

################################################################


################################################################
# The model we used for sentence embedding transfermation
def model_generator(model_name):
    model_embedding = SentenceTransformer(model_name)

    return model_embedding

def max_candidate(label_,sent_embedding_,label_embeddings):
    # return the max similarity candidates
    #output: [[similarity, synset, the pos of synset]]
    max_nodeid = [0,""]
    
    if label_ not in label_embeddings:
        # label is not exists in the labels embeddings means:
            #there is no sysets for this label
            # return ""
        pass
    else:
        # label exists
        # return the max similarity candidates
        for candit, embedding in label_embeddings[label_]:
            # cosine similarity
            similar = rltk.cosine_similarity(list(embedding), list(sent_embedding_))
            temp = [similar, candit]
            if temp > max_nodeid:
                # if temp is lager than max_nodeid, swap the max_nodeid
                
                max_nodeid = temp
            
    return max_nodeid[1]

def sort_candidate(label_,sent_embedding_,label_embeddings):
    # node id is sorted by the similar, desc
    #output: [[similarity, synset, the pos of synset]]
    sort_nodeid = []
    
    if label_ not in label_embeddings:
        pass
    else:
        i = 0
        for candit, embedding in label_embeddings[label_]:
            similar = - rltk.cosine_similarity(list(embedding), list(sent_embedding_))
            
            temp = [similar, candit,i]
            
            heapq.heappush(sort_nodeid, temp)
            i +=1
            
    return sort_nodeid

def generate_idx_combine(len_nodeid1,len_nodeid2):
    # generate index combination for nodeId combination.
    # nodeId combination will be used to check whther the prediction limitation satisfied
    
    total = len_nodeid1 + len_nodeid2

    for i in range(total):
        for len1 in range(min(i+1,len_nodeid1)):
            len2 = i-len1
            if len2 < len_nodeid2:
                yield len1, len2

    

def sentence_embedding(wn_gold, model, label_embeddings = None, word2sentence = word2sentence, word2limit= word2limit):
    # sentence-transformer-bert Calculation
    wn_predict = []
    # frequency place of output data
    freq = []

    sents_combine = []
    for line in wn_gold:
        sentence = line_sentence(line, word2sentence)
        sents_combine.append(sentence)
    sents_embedding = model.encode(sents_combine)

    for line,sent_embedding in zip(wn_gold,sents_embedding):
        label1 = line[0]
        label2 = line[2]
        predicate = line[1]

        #obtain the max similar item for label1
        sort_nodeid1 = sort_candidate(label1,sent_embedding,label_embeddings)
        
        #obtain the max similar item for label2
        sort_nodeid2 = sort_candidate(label2,sent_embedding,label_embeddings)
        
        node_id1 = ""
        node_id2 = ""
        freq1 = -1
        freq2 = -1
        for idx1, idx2 in generate_idx_combine(len(sort_nodeid1),len(sort_nodeid2)):

            _, node_id1, freq1 = sort_nodeid1[idx1]

            _, node_id2, freq2 = sort_nodeid2[idx2]

            status = predicate_limitation_check(node_id1, node_id2, predicate, word2limit)

            if status:
                break
            else:
                continue
            
                
        wn_predict.append([label1, line[1], label2,node_id1,node_id2])
        freq.append(tuple([freq1, freq2]))
    #print(freq)
        
    return wn_predict, freq

################################################################
    