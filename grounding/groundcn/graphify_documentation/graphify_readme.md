# GRAPHIFY

To turn a sentence or sentences into a graph, we using `graphify.py`. 

To use the function in your own script:
```Python
from graphify import graphify

sentence = "John is waiting for his car to be finished."
graph = graphify(sentence)
```

## Graph Structure
The graph returned by `graphify()` is a dictionary with the following structure:
```Python
graph = {'edges': {dictionary of edges},
	 'nodes': {dictionary of nodes},
	 'sentence': 'John is waiting for his car to be finished.',
	 'tokenized_sentence': ['John', 'is', 'waiting', 'for', 'his', 'car', 'to', 'be', 'finished', '.']
```

Each node has a unique id that is used to index it in `graph['nodes']`. A node is represented as a dictionary:
```Python
graph['nodes'][node_id] = {'phrase': A sublist of the tokenized sentence that this node represents,
			   'start_idx': The position in the tokenized sentence where the phrase begins,
			   'end_idx': The position in the tokenized sentence where the phrase ends,
			   'entity_type': A list that indicates the entity type of each word in the phrase}
```

Each edge has a unique id that is used to index it in `graph['edges']`. An edge is represented as a dictionary:
```Python
graph['edges'][edge_id] = {'head_node_id': The id of the head node (since this is a directed graph),
			   'end_node_id': The id of the tail node,
			   'edge_name': The edge name can either be the semantic role connecting the 
			   		predicate to its argument or a coref edge (marked 'coref') if 
			   		it is connecting coreferential mentions. 
			   		It can also be 'sub', which we cover later.}
```

## How the Graph is Created
Let's first visualize the **final** graph for the sentence above and then go over how we arrive there.

<img src="imgs/graph.png" width="461" height="314" />

**The creation of the graph happens in three steps.**

1. **Semantic Role Labeling (SRL)**<br/><br/>
The graph is first created by passing the sentence(s) through a semantic role labeling parser. From this parse, we create nodes for the arguments and predicates and edges between the two. The edge names are the roles.  <br/><br/>
Creating this graph can be tricky because SRL can return multiple predicates per sentence. This means that sometimes, a node will contain a phrase that is a subphrase of another node. Notice in the graph above how **for his car to be finished** is an argument of **waiting** while **his car** is an argument of **finished**. To address, this, we add an edge with name **sub** for subphrases. The graph looks like this after SRL.

<img src="imgs/graph_srl.png" width="461" height="314" />

2. **Coreference Resolution**<br/><br/>
We then add edges to the graph based on coreference resolution. This allows us to not only connect mentions in the same sentence, but also across sentences if we are feeding in a string with multiple sentences into `graphify()`. <br/><br/>
In this case, we want to connect **his** to **John**. However, **his** is not in a node, but **his** appears as a subphrase to the node **his car**. Similar to above, we create a **his** node, connect it to **his car** using a **sub** edge, and then connect **his** to **John** using a **coref** edge. Coreference resolution edges also allow us to connect entities across sentences if we are feeding in multiple sentences into `graphify()`. The graph structure is now complete and looks like the first graph above.

3. **Named Entity Recognition (NER)**<br/><br/>
For each node, we mark (as metadata) whether each word is a named entity or not. This doesn't affect the graph structure, but does allow some flexibility in linking to a KB. For example, maybe we don't want to link names like **James**.