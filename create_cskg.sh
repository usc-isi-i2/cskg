# Create CSKG

## Extract individual graphs

### ATOMIC
kgtk import_atomic input/v4_atomic_all_agg.csv > tmp/kgtk_atomic.tsv 

### ConceptNet
kgtk import_conceptnet --english_only input/conceptnet-assertions-5.7.0.csv > tmp/kgtk_conceptnet.tsv

### ROGET
kgtk import_concept_pairs -i input/antonyms.txt --source RG --relation /r/Antonym > tmp/kgtk_roget_antonyms.tsv
kgtk import_concept_pairs -i input/synonyms.txt --source RG --relation /r/Synonym > tmp/kgtk_roget_synonyms.tsv

### Visual Genome
kgtk import-visualgenome -i input/visualgenome/scene_graphs.json --attr-synsets input/visualgenome/attribute_synsets.json > tmp/kgtk_visualgenome.tsv

### WordNet
kgtk import_wordnet > tmp/kgtk_wordnet.tsv

## Combine sources and add IDs
kgtk cat tmp/kgtk_atomic.tsv tmp/kgtk_conceptnet.tsv tmp/kgtk_roget_synonyms.tsv tmp/kgtk_roget_antonyms.tsv tmp/kgtk_wordnet.tsv tmp/kgtk_visualgenome.tsv tmp/wikidata20200504/kgtk_wikidata.tsv / sort -c 'node1,relation,node2' / add_id --id-style node1-label-node2-num / reorder_columns --columns id ... > output/cskg_base.tsv

## Compact the graph
kgtk compact -i output/cskg_base.tsv -o output/cskg_compact.tsv --columns node1 relation node2 --presorted False --compact-id True --build-id --overwrite-id

## Concatenate mappings
kgtk cat output/cskg_compact.tsv tmp/mapping_wn_wn.tsv tmp/lexical_mappings.tsv > output/cskg_compact_with_mappings.tsv

## Concatenate CSKG with the mappings and deduplicate
kgtk connected_components --properties mw:SameAs      --input-file output/cskg_compact_with_mappings.tsv      / lift --columns-to-lift node1 node2 --lift-suffix=      --input-file output/cskg_compact_with_mappings.tsv      --label-file -      --label-select-value connected_component      / filter  --invert -p ';mw:SameAs;'      / compact --output-file output/cskg_connected.tsv --columns node1 relation node2 --compact-id --overwrite-id True --build-id --overwrite-id


# Working with CSKG

## Compute statistics
kgtk graph_statistics -i cskg.tsv --directed --degrees --hits --pagerank --statistics-only --log summary.txt

## Compute embeddings
kgtk unlift node1;label node2;label -i cskg.tsv / \
sort -c / text_embedding \
            --debug --embedding-projector-metadata-path none \
            --embedding-projector-metadata-path none \
            --label-properties "label" \
            --isa-properties "/r/IsA" \
            --description-properties "/r/DefinedAs" \
            --property-value "/r/Causes" "/r/UsedFor" \
            --has-properties "" \
            -f kgtk_format \
            --output-format kgtk_format \
            --use-cache \
            --model bert-large-nli-cls-token \
            > cskg_embedings.txt

## Compute paths
kgtk paths --max_hops 2 --path_file path_nodes.tsv -i cskg.tsv --statistics_only --directed > paths.tsv
