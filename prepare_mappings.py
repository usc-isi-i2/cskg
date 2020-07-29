
mapping_file='input/mappings/wn_wn_mappings.csv'

def print_edge(cols):
    return '\t'.join(cols) + '\n'

rows=[]
with open('tmp/kgtk_mapping_wn_wn.tsv', 'w') as w:
    columns=['node1', 'relation', 'node2', 'node1;label', 'node2;label','relation;label', 'relation;dimension', 'weight', 'source', 'origin', 'sentence', 'question']
    w.write(print_edge(columns))
    with open(mapping_file, 'r') as f:
        header=next(f)
        for line in f:
            data=line.split('\t')
            new_row=[*data[:3], "", "", "", "", "", "", "", "", ""]
            w.write(print_edge(new_row))
