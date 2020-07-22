import csv
import os
import random
from utils import extract_node_data, normalize_relation
import pandas as pd
VERSION="004"

from relation_mapping import COMMONSENSE_MAPPING

def write_data(l, f):
    with open(f, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        for line in l:
            spamwriter.writerow(line)

def camel_case_split(s): 
    s=s.strip()
    start_idx = [i for i, e in enumerate(s) 
                 if e.isupper()] + [len(s)] 
  
    start_idx = [0] + start_idx 
    return [s[x: y] for x, y in zip(start_idx, start_idx[1:])] 

dataset='cskg'#cn-original'

edges_file='../output_v%s/%s/edges.tsv' % (VERSION, dataset)
nodes_file=edges_file.replace('edges', 'nodes')
output_dir='../output_v%s/emb_data/%s' % (VERSION, dataset)

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

node2label, node2pos = extract_node_data(nodes_file)
rows=[]
with open(edges_file, 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter='\t')
	cols=reader.__next__()
	cols=cols[:3]
	for row in reader:
		my_row=row[:3]
		if my_row[2] in node2label.keys():
			my_row[2]=node2label[my_row[2]]

			norm_rel=normalize_relation(my_row[1])
			if norm_rel in COMMONSENSE_MAPPING.keys():
				my_row[1]=COMMONSENSE_MAPPING[norm_rel]
				#my_row[1]=' '.join(camel_case_split(norm_rel)).lower().strip()
				if my_row and my_row[0].strip() and my_row[2].strip():
					rows.append(my_row)

for n, l in node2label.items():
	row=[n, 'label', l]
	rows.append(row)

df=pd.DataFrame(rows, columns=cols)

df.sort_values(by=cols).to_csv( '%s/text_emb.tsv' % output_dir, index=False, sep='\t')

#write_data(rows, '%s/text_emb.tsv' % output_dir)

