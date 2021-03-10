# Representation

CSKG is modeled as a **hyper-relational graph**. It describes edges in a tabular format, following the KGTK [data model](https://kgtk.readthedocs.io/en/latest/data_model/) and [file specification](https://kgtk.readthedocs.io/en/latest/specification/).

The edges in CSKG are described by ten columns:

Following KGTK, the primary information about an edge consists of its `id`, `node1`, `relation`, and `node2` (**default edge columns**). 

Next, we include four **lifted edge columns**, using KGTK's abbreviated way of representing triples about the primary elements, such as `node1;label` or `relation;label` (label of `node1` and of `relation`). 

Each edge is completed by two qualifiers (**secondary edges**): `source`, which specifies the source(s) of the edge (e.g., "CN" for ConceptNet), and `sentence`, containing the linguistic lexicalization of a triple, if given by the original source. 

Summarizing, here are the 10 columns that comprise the CSKG edge representation: 
1. `id` is an edge identifier, constructed by concatenating its node1, relation, and node2 elements. We aim to have edge ids be consistent across CSKG versions.
2. `node1` is a node identifier, must have a single value, cannot be empty, cannot have empty spaces. 
3. `relation` is an identifier, must have a single value from a predefined list, cannot be empty, cannot have empty spaces.
4. `node2` is a node identifier, must have a single value, cannot be empty, cannot have empty spaces. 
5. `node1;label` is a textual label for `node1`. It can have multiple different values, separated with a "|" character. Can be empty.
6. `node2;label` is a textual label for `node2`. It can have multiple different values, separated with a "|" character. Can be empty.
7. `relation;label` is a textual label for `relation`. It can have multiple different values, separated with a "|" character. Can be empty.
8. `relation;dimension` is an abstract knowledge type for a relation (e.g., "spatial"), one of the predefined 13 categories in [this paper](https://arxiv.org/abs/2101.04640). Can have multiple values. Can be empty.
9. `source` is a list of the source KGs in which this edge was found (e.g., ConceptNet). Can have multiple values, separated by "|". Can be empty.
10. `sentence` is the original sentence from which the triple was derived. Can have multiple values, separated by "|" (in case we have multiple sources). Can be empty.


CSKG is mainly described with a single tabular file. Auxiliary KGTK files can be added to describe additional knowledge about some edges, such as their weight, through the corresponding edge `id`s. 
