import pandas as pd
import json

filename="../input/atomic/v4_atomic_all_agg.csv"

edges_file='../output_v004/atomic/edges_v004.csv'
nodes_file='../output_v004/atomic/nodes_v004.csv'

df = pd.read_csv(filename,index_col=0)
df.iloc[:,:9] = df.iloc[:,:9].apply(lambda col: col.apply(json.loads))

df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)

print(df.columns)

print(len(df))

datasource='atomic'
weight=1.0
other={}

edges_cols=['subject', 'predicate', 'object', 'datasource', 'weight', 'other']
nodes_cols=['id', 'label', 'aliases', 'pos', 'datasource', 'other']

def make_node(x):
    und_x=x.replace(' ', '_')
    pref_und_x='at:%s' % und_x
    return pref_und_x

columns=df.columns

my_rows=[]
node_rows=[]
all_nodes=set()

counter=0
for event, row in df.iterrows():
	e=event.replace('PersonX', '').strip()
	e=e.replace('PersonY', '').strip()
	e=e.replace('the ___', '')
	e=e.replace('___', '')
	e=e.replace("'s", '')
	while '  ' in e:
		e=e.replace('  ', ' ')
	e=e.strip()
	for c in columns:
		for v in row[c]:
			if v=='none': continue
			v=v.rstrip('.').replace('to Y', '').lower()
			v=v.replace('personx', '').replace('persony', '').replace('person x', '').replace('person y', '').replace("'s", '').replace('  ', ' ').strip()
			while '  ' in v:
				v=v.replace('  ', ' ')
			v=v.strip()
			n1=make_node(e)
			n2=make_node(v)
			my_rows.append([n1, make_node(c), n2, datasource, weight, other])

			if n1 not in all_nodes:
				row1=[n1, e, '', '', datasource, other]
				node_rows.append(row1)
				all_nodes.add(n1)
			if n2 not in all_nodes:
				row2=[n2, v, '', '', datasource, other]
				node_rows.append(row2)
				all_nodes.add(n2)

			#print(e, 'at:%s' % c, v)
			counter+=1
print(counter)

nodes_df=pd.DataFrame(node_rows, columns=nodes_cols)
nodes_df.sort_values('id').to_csv(nodes_file, index=False, sep='\t')

edges_df=pd.DataFrame(my_rows, columns=edges_cols)
edges_df.sort_values(by=['subject', 'predicate','object']).to_csv(edges_file, index=False, sep='\t')
