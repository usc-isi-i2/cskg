# Version
VERSION="004"

# CSKG columns
nodes_cols=['id', 'label', 'aliases', 'pos', 'datasource', 'other']
edges_cols=['subject', 'predicate', 'object', 'datasource', 'weight', 'other']

# CUSTOM DATA WE ADD
custom_dataset='d/mowgli'

# NAMESPACES
mowgli_ns='mw'
visualgenome_ns='vg'
wordnet_ns='wn'
wdt_ns='wd'

# DATASOURCES
cn_ds='CN'
vg_ds='VG'
wn_ds='WN'
wdt_ds='WDT'
fn_ds='FN'
mw_ds='MOWGLI'

# RELATIONS
has_pos='PartOfSpeech'
has_pos_form='POSForm'
is_pos_form_of='IsPOSFormOf'
wordnet_sense='OMWordnetOffset'

pwordnet_sense='PWordnetSynset'
subject='Subject'
objct='Object'
in_image='InImage'
has_prop='/r/HasProperty'
sameas='SameAs'

# ConceptNet rel-s
symmetric_rels=["/r/Antonym", "/r/DistinctFrom", "/r/EtymologicallyRelatedTo", "/r/LocatedNear", "/r/RelatedTo", "/r/SimilarTo", "/r/Synonym"]

# POS mappings
pos_mapping={'a': 'Adjective', 'n': 'Noun', 'r': 'Adverb', 'v': 'Verb'}
