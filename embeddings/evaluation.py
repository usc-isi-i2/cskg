import numpy as np
import json
from sklearn.metrics import ndcg_score


def load_cue_targets(input_file):
    """
    Load cur and targets from  json file 

    Parameters
    ----------
    input_file: str
        file path of cue_targets json file 

    Return
    ----------
    cue_targets: dict
        A dictionary whose key is the cue/label, value is a list of the similar targets to the cue/label

    """
    cue_targets = {}
    with open(input_file) as f:
        cue_targets = json.load(f)
    
    return cue_targets

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


## may have some issues...
## TODO... implement bymyself
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