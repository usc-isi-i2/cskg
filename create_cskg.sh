# Create CSKG

## Extract individual graphs

### ATOMIC
kgtk import_atomic v4_atomic_all_agg.csv > kgtk_atomic.tsv 

### ConceptNet
kgtk import_conceptnet --english_only conceptnet-assertions.csv > kgtk_conceptnet.tsv

### ROGET
kgtk import_concept_pairs -i antonyms.txt --source RG --relation /r/Antonym > kgtk_roget.tsv
kgtk import_concept_pairs -i synonyms.txt --source RG --relation /r/Synonym >> kgtk_roget.tsv

### Visual Genome
kgtk import-visualgenome -i scene_graphs.json --attr-synsets attribute_synsets.json > kgtk_visualgenome.tsv

### WordNet
kgtk import_wordnet > kgtk_wordnet.tsv

## Combine sources and add IDs
kgtk cat kgtk_atomic.tsv kgtk_conceptnet.tsv kgtk_roget_synonyms.tsv kgtk_roget_antonyms.tsv kgtk_wordnet.tsv kgtk_visualgenome.tsv / sort -c 'node1,relation,node2' / add_id --id-style node1-label-num / reorder_columns --columns id ... > cskg_base.tsv

## Compact the graph
kgtk compact -i output/cskg_base.tsv -o output/cskg_compact.tsv --columns node1 relation node2 --presorted False --compact-id True --build-id --overwrite-id

## Concatenate CSKG with the mappings and deduplicate
kgtk cat cskg_compact.tsv mapping1.tsv ... mapping6.tsv / merge --relation mw:sameAs / sort -c 'node1,relation,node2' > cskg.tsv

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
