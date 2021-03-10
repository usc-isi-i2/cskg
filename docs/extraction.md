# Extraction

CSKG integrates seven sources, selected based on their popularity in existing QA work: a commonsense knowledge graph ConceptNet, a visual commonsense source Visual Genome, a procedural source ATOMIC, a general-domain source Wikidata, and three lexical sources, WordNet, Roget, and FrameNet.

The integration consists of three key steps: adaptations per source, mappings between sources, and refinements. 

The entire procedure of importing the individual sources and consolidating them into CSKG is implemented with KGTK operations, and can be found as a single Bash script [here](https://github.com/usc-isi-i2/cskg/blob/master/consolidation/create\_cskg.sh). 

## 1) Adaptations and modeling decisions per source
**ConceptNet** We keep the original edges of ConceptNet 5.7 expressed with 47 relations in total. 

**ATOMIC** We also include the entire ATOMIC KG, preserving the original nodes and its nine relations. To enhance lexical matching between ATOMIC and other sources, we add normalized labels of its nodes, e.g., adding a second label "accepts invitation" to the original one "personX accepts personY's invitation".

**FrameNet** We import four node types from FrameNet: frames, frame elements (FEs), lexical units (LUs), and semantic types (STs), and we reuse 5 categories of FrameNet edges: frame-frame (13 edge types), frame-FE (1 edge type), frame-LU (1 edge type), FE-ST (1 edge type), and ST-ST (3 edge types). Following principle 2 on edge type reuse, we map these 19 edge types to 9 relations in ConceptNet, e.g., `is_causative_of` is converted to `/r/Causes`. 

**Roget** We include all synonyms and antonyms between words in Roget, by reusing the ConceptNet relations `/r/Synonym` and `/r/Antonym` (Principle 2).

**Visual Genome** We represent Visual Genome as a KG, by representing its image objects as WordNet synsets (e.g., `wn:shoe.n.01`). We express relationships between objects via ConceptNet's `/r/LocatedNear` edge type. Object attributes are represented by different edge types, conditioned on their part-of-speech: we reuse ConceptNet's `/r/CapableOf` for verbs, while we introduce a new relation `mw:MayHaveProperty` for adjective attributes.

**Wikidata** We include the *Wikidata-CS* subset [(Ilievski et al., 2020)](https://arxiv.org/abs/2008.08114). Its 101k statements have been manually mapped to 15 ConceptNet relations.

**WordNet** We include four relations from WordNet v3.0 by mapping them to three ConceptNet relations: hypernymy (using `/r/IsA`), part and member holonymy (through `/r/PartOf`), and substance meronymy (with `/r/MadeOf`). 


## 2) Mappings between sources

We perform node resolution by applying existing identity mappings (principle 3) and generating probabilistic mappings automatically (principle 4). 
We introduce a dedicated relation, `mw:SameAs`, to indicate identity between two nodes.

**WordNet-WordNet** The WordNet v3.1 identifiers in ConceptNet and the WordNet v3.0 synsets from Visual Genome are aligned by leveraging ILI: [the WordNet InterLingual Index](https://github.com/globalwordnet/ili) which generates 117,097 `mw:SameAs` mappings.

**WordNet-Wikidata** We generate links between WordNet synsets and Wikidata nodes as follows.
For each synset, we retrieve 50 candidate nodes from a customized index of Wikidata. 
Then, we compute sentence embeddings of the descriptions of the synset and each of the Wikidata candidates by using a pre-trained XLNet model. 
We create a `mw:SameAs` edge between the synset and the Wikidata candidate with highest cosine similarity of their embeddings.
Each mapping is validated by one student. In total, 17 students took part in this validation. Out of the 112k edges produced by the algorithm, the manual validation marked 57,145 as correct. We keep these in CSKG and discard the rest.

**FrameNet-ConceptNet** We link FrameNet nodes to ConceptNet in two ways. FrameNet LUs are mapped to ConceptNet nodes through the Predicate Matrix with 3,016 `mw:SameAs` edges. Then, we use 200k hand-labeled sentences from the FrameNet corpus, each annotated with a target frame, a set of FEs, and their associated words. We treat these words as LUs of the corresponding FE, and ground them to ConceptNet with the rule-based method of KagNet.

**Lexical matching** We establish 74,259 `mw:SameAs` links between nodes in ATOMIC, ConceptNet, and Roget by exact lexical match of their labels. We restrict this matching to lexical nodes (e.g., `/c/en/cat` and not `/c/en/cat/n/wn/animal`).

## 3) Deduplication and other refinements

After transforming each source to the representation described here, we concatenate the sources in a single graph. We deduplicate this graph and append all mappings, resulting in `CSKG*`. Finally, we apply the mappings to merge identical nodes (connected with `mw:SameAs`) and perform a final deduplication of the edges, resulting in our consolidated `CSKG` graph. 
