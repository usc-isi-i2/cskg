import argparse
import hashlib
import itertools
from json import dumps

from allennlp.predictors.predictor import Predictor
import spacy
from spacy.cli.download import download as spacy_download
from tqdm import tqdm


# Change this to use the GPU
CUDA_DEVICE = -1
COREF_MODEL = "https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz"
SRL_MODEL = "https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz"
SPACY_MODEL = "en_core_web_lg"

def create_node(phrase: list, start_idx: int, end_idx: int, entity_type: list = None):
    """


    """
    if entity_type == None:
        entity_type = [None]*len(phrase)
    else:
        assert len(entity_type) == len(phrase)

    node = {'phrase': phrase,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'entity_type': entity_type}

    # For the node id, turn the node dictionary into a string, sort the string,
    # and get the hash of the MD5 hash of the string.
    result = hashlib.md5(''.join(sorted(node.__repr__())).encode())
    node_id = result.hexdigest()
    return node, node_id

def create_edge(head_node_id: str, tail_node_id: str, edge_name: str, edge_source: str):
    """

    """
    edge = {'head_node_id': head_node_id,
            'tail_node_id': tail_node_id,
            'edge_name': edge_name,
            'edge_source': edge_source}

    # For the edge id, turn the edge dictionary into a string, sort the string,
    # and get the hash of the MD5 hash of the string.
    result = hashlib.md5(''.join(sorted(edge.__repr__())).encode())
    edge_id = result.hexdigest()

    return edge, edge_id

def create_nodes_and_edges_from_srl_dict(srl_dict, all_words):
    """ Creates nodes and edges from a predicate and its arguments


    """
    all_tags = srl_dict['tags']
    nodes = {}
    edges = {}

    # Find the predicate phrase
    predicate_phrase = []
    start_idx = None

    for idx, (word, tag) in enumerate(zip(all_words, all_tags)):
        # If we find the start of the predicate, add it to the list
        if tag == 'B-V':
            predicate_phrase.append(word)
            start_idx = idx
        elif start_idx != None:
            predicate_phrase.append(word)

        # If we are at the end of the list or the next word is not a continuation, break.
        # We have found the last token of the predicate.
        if (len(all_words)-1 == idx or 'I-' not in all_tags[idx+1]) and start_idx != None:
            break

    # If we weren't able to find a predicate phrase, then return empty nodes and edges
    # Found when input is `hardworking`.
    if predicate_phrase == []:
        return {}, {}

    # Create a node from the predicate phrase.
    predicate_node, predicate_node_id = create_node(predicate_phrase, start_idx, idx)
    nodes[predicate_node_id] = predicate_node

    # Create nodes for the arguments associated with the predicate
    # and an edge between the argument nodes to the predicate node
    phrase = []
    start_idx = None

    for idx, (word, tag) in enumerate(zip(all_words, all_tags)):
        # If we find the start of the argment, add it to the phrase list.
        if 'B-' in tag and tag != 'B-V':
            phrase.append(word)
            start_idx = idx
        # If `start_idx` is not `None`, the current word is a continuation of a phrase.
        elif start_idx != None:
            phrase.append(word)

        # If we are at the end of the words or the next word is not a
        # continuation, we have found the last token of the current argument.
        #
        # Create a node with `phrase`.
        # Create an edge between this phrase node and the predicate node.
        if (len(all_words)-1 == idx or 'I-' not in all_tags[idx+1]) and start_idx != None:
            node, node_id = create_node(phrase, start_idx, idx)
            edge, edge_id = create_edge(predicate_node_id, node_id, edge_name=tag[2:], edge_source='srl')
            nodes[node_id] = node
            edges[edge_id] = edge

            # Reset `start_idx` and `phrase` list
            start_idx = None
            phrase = []

    # Sometimes, a SRL parse will return a predicate with no arguemnts.
    # In this case, skip this predicate.
    if edges == {}:
        return {}, {}
    else:
        return nodes, edges

def create_graph_from_srl_parse(sentence: str):
    out = srl_predictor.predict(sentence)
    tokenized_sentence = out['words']
    nodes = {}
    edges = {}

    ### Initialize graph using SRL parser
    # Each sentence may have multiple predicates. AllenNLP SRL parser
    # returns a dictionary for each predicate and its associated argument.
    # Iterate through the dictionaries, creating nodes and edges.
    for srl_dict in out['verbs']:
        cur_nodes, cur_edges = create_nodes_and_edges_from_srl_dict(srl_dict, out['words'])
        nodes.update(cur_nodes)
        edges.update(cur_edges)

    # Resolve subparses. Some phrases may be subphrases of a longer phrase.
    # For these, add an edge from the longer phrase to the shorter one with the label 'sub'
    for node1_id, node2_id in itertools.combinations(nodes.keys(), 2):
        node1_spans = (nodes[node1_id]['start_idx'], nodes[node1_id]['end_idx'])
        node2_spans = (nodes[node2_id]['start_idx'], nodes[node2_id]['end_idx'])

        if node1_spans[0] >= node2_spans[0] and node1_spans[1] <= node2_spans[1] and node1_spans != node2_spans:
            edge, edge_id = create_edge(node2_id, node1_id, 'sub', 'srl')
            edges[edge_id] = edge
        elif node2_spans[0] >= node1_spans[0] and node2_spans[1] <= node1_spans[1] and node1_spans != node2_spans:
            edge, edge_id = create_edge(node1_id, node2_id, 'sub', 'srl')
            edges[edge_id] = edge

    # If we weren't able to create any nodes via the SRL parse, then create a node with the `sentence` tokenized
    # This can happen for several reasons:
    # If SRL cannot parse (e.g. when `sentence` is a short phrase)
    # If SRL finds predicates with no arguments.
    # If SRL finds arguments with no predicates.
    if nodes == {}:
        # If there aren't any nodes, there shouldn't be any edges
        assert edges == {}
        node, node_id = create_node(tokenized_sentence, start_idx=0, end_idx=len(tokenized_sentence)-1)
        nodes[node_id] = node

    return tokenized_sentence, nodes, edges

def add_entity_types_to_graph(sentence, nodes, edges):
    entities = []
    for word in spacy_parser(sentence):
        if word.ent_type_:
            for node_id in nodes:
            	for idx, node_word in enumerate(nodes[node_id]['phrase']):
            		if node_word.lower() == word.text.lower():
            			nodes[node_id]['entity_type'][idx] = word.ent_type_

    return nodes, edges

def get_coreference_node(nodes, edges, root_node_ids, indices):
    """ Checks if a phrase exists as a node. If not, we find
    the node that this phrase can be a subphrase for and create a node this way.
    """
    start_idx, end_idx = indices

    for node_id, node in nodes.items():
        if node_id not in root_node_ids:
            continue

        if start_idx >= node['start_idx'] and end_idx <= node['end_idx']:
            if start_idx == node['start_idx'] and end_idx == node['end_idx']:
            	return nodes, edges, node_id

            # Create a new node that is a subset of the current one
            else:
            	phrase_start_idx = start_idx - node['start_idx']
            	phrase_end_idx = phrase_start_idx + end_idx - start_idx
            	new_node, new_node_id = create_node(node['phrase'][phrase_start_idx:phrase_end_idx+1],
            										start_idx, end_idx,
            										node['entity_type'][phrase_start_idx:phrase_end_idx+1])

            	# Add edge between the head node and the new node
            	new_edge, new_edge_id = create_edge(node_id, new_node_id, edge_name='sub', edge_source='coref')
            	nodes[new_node_id] = new_node
            	edges[new_edge_id] = new_edge

            	return nodes, edges, new_node_id

    # print('Could not create coref edge')
    return nodes, edges, None

def add_coreference_edges_to_graph(sentence, tokenized_sentence, nodes, edges):
    # When tokenizing input, coref model does not strip off consecutive spaces.
    # This can result in the tokenized output of the coref model having a space as a token.
    # Address this here by splitting and rejoining, removing consecutive spaces.
    sentence = ' '.join(sentence.split())

    # First get a list of root nodes.
    # Coreference edges will only be added to the root nodes in the SRL graph.
    root_node_ids = list(nodes.keys())
    for edge in edges.values():
        head_node_id = edge['head_node_id']
        if head_node_id in root_node_ids:
            root_node_ids.remove(head_node_id)

    try:
        output = coref_predictor.predict(sentence)
    except:
        # print('Coreference model throws error on "', sentence, '"')
        return nodes, edges

    # Check that the tokenziation returned by SRL matches the tokenization returned by coreference
    assert tokenized_sentence == output['document']

    # Add clusters as edges to the graph
    clusters = output['clusters']
    for cluster in clusters:
        antecedant_indices = cluster[0]
        antecedant = tokenized_sentence[antecedant_indices[0]:antecedant_indices[1]+1]
        nodes, edges, antecedant_node_id = get_coreference_node(nodes, edges, root_node_ids, antecedant_indices)

        if antecedant_node_id == None:
            continue

        for coindexed_indices in cluster[1:]:
            nodes, edges, coindexed_node_id = get_coreference_node(nodes, edges, root_node_ids, coindexed_indices)

            if coindexed_indices == None:
            	continue

            coref_edge, coref_edge_id = create_edge(antecedant_node_id, coindexed_node_id, edge_name='coref', edge_source='coref')
            edges[coref_edge_id] = coref_edge

    return nodes, edges

def graphify(sentence: str):
    tokenized_sentence, nodes, edges = create_graph_from_srl_parse(sentence)

    nodes, edges = add_entity_types_to_graph(sentence, nodes, edges)

    nodes, edges = add_coreference_edges_to_graph(sentence, tokenized_sentence, nodes, edges)

    ### Create graph dictionary and return it
    graph = {'sentence': sentence,
             'tokenized_sentence': tokenized_sentence,
             'nodes': nodes,
             'edges': edges}
    return graph

def graphify_dataset(sentences, output_file=None):
    global spacy_parser, coref_predictor, srl_predictor

    try:
        spacy_parser = spacy.load(SPACY_MODEL, disable=['parser', 'tagger'])
    except IOError:
        spacy_download(SPACY_MODEL)
        spacy_parser = spacy.load(SPACY_MODEL, disable=['parser', 'tagger'])

    coref_predictor = Predictor.from_path(COREF_MODEL, cuda_device=CUDA_DEVICE)
    srl_predictor = Predictor.from_path(SRL_MODEL, cuda_device=CUDA_DEVICE)

    graphs=[]
    if output_file:
            writer = open(output_file, 'w')

    for sentence in tqdm(sentences):
            graph = graphify(sentence.strip())
            graphs.append(graph)
            if output_file:
            		writer.write(dumps(graph) + '\n')

    if output_file:
            writer.close()
    return graphs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, \
        help='Path to input text file. Each line in the file will be turned into a graph.')
    parser.add_argument('--output', type=str, \
        help='Path to output JSONL file. Each line in the output file will be a graph corresponding to a line in the input file')
    args = parser.parse_args()

    # Graphify the input file
    with open(args.input) as f:
        sentences=[]
        for sentence in f:
            	sentences.append(sentence)
        graphs=graphify_dataset(sentences, args.output)

if __name__ == '__main__':
    main()
