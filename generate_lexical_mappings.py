from collections import defaultdict

def check_source(s, sources):
    if '|' in s:
        for source in s.split('|'):
            if source in sources: return True
        return False
    else:
        return s in sources

def lexical_node(node):
    if node.startswith('/c/en/') and len(node.split('/'))>4:
        return False
    else:
        return True

input_file='output/cskg_compact.tsv'
output_file='tmp/lexical_mappings.tsv'

sources=['AT', 'RG', 'CN']
identity_rel='mw:SameAs'

lbl2ids=defaultdict(set)

with open(input_file, 'r') as f:
    header=next(f)
    for line in f:
        data=line.split('\t')
        if check_source(data[8], sources):
            node1=data[1]
            node2=data[3]
            if lexical_node(node1):
                node1_label=data[4]
                lbl2ids[node1_label].add(node1)
            if lexical_node(node2):
                node2_label=data[5]
                lbl2ids[node2_label].add(node2)
print(len(lbl2ids))

with open('tmp/lexical_mappings.tsv', 'w') as w:
    w.write(header)
    for label, ids in lbl2ids.items():
        if len(ids)<=1: continue

        list_ids=list(ids)
        for i in range(len(list_ids)-1):
            edge_id='%s-%s-%s-1' % (list_ids[i], identity_rel, list_ids[i+1])
            row=[edge_id, list_ids[i], identity_rel, list_ids[i+1], '', '', '', '', 'LEX', '']
            w.write('\t'.join(row) + '\n')
