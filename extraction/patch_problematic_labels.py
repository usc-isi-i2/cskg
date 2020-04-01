import config
import json
import pandas as pd

VERSION=config.VERSION
cskg_dir='../output_v%s/cskg_ready' % VERSION
output_merged_dir='../output_v%s/cskg_ready' % VERSION

cskg_nodes_file='%s/nodes_v%s.csv' % (cskg_dir, VERSION)
merged_nodes_file='%s/nodes2_v%s.csv' % (output_merged_dir, VERSION)


prob_json='../input/problematic.json'

with open(prob_json, 'r') as f:
	problematic_nodes=json.load(f)

print(len(problematic_nodes))

from collections import defaultdict
import csv

mapping={}
startswith=defaultdict(int)
for rawk in problematic_nodes:
	rawk=rawk.strip()
	k=rawk.split('\t')[0]

	node_id=k
	startswith[k[:3]]+=1
	if k[:4]!='vg:I' and not (k[:3]=='wd:' and '+' not in k):
		if k[0]=='"':
			k=k.replace('"', '')
		ind_nodes=k.split('+')
		cn_in=False
		wn_in=False
		vg_in=False
		for n in ind_nodes:
			if n[:3]=='/c/':
				cn_in=True
				cn_label=n[6:].split('/')[0].replace('_', ' ').strip()
			if n[:3]=='wn:':
				wn_in=True
				wn_label=n[3:].split('.')[0].replace('_', ' ').strip()
			if n[:3]=='vg:':
				vg_in=True
				vg_label=n[3:].replace('_', ' ').strip()
		if cn_in:
			mapping[node_id]=cn_label
		elif wn_in:
			mapping[node_id]=wn_label
		elif vg_in:
			mapping[node_id]=vg_label
		else:
			print(k.strip(), 'RAW:', rawk)

with open(cskg_nodes_file, 'r') as csvfile:
	reader=csv.reader(csvfile, delimiter='\t', quotechar='"')
	new_rows=[]
	c=0
	headers = next(reader, None)
	new_rows.append(headers)
	for row in reader:
		if row[0] in mapping.keys():
			row[1]=mapping[row[0]]
			c+=1
		new_rows.append(row)

	print(startswith)
	print(c, 'changes')


	nodes_df=pd.DataFrame(new_rows, columns=headers)
	nodes_df.sort_values('id').to_csv(merged_nodes_file, index=False, sep='\t')
