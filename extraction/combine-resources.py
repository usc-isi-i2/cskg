import config
import json
import pandas as pd
import os

VERSION=config.VERSION
output_dir='../output_v%s/cskg' % VERSION
data_dir='../output_v%s' % VERSION

cn_nodes_file='%s/conceptnet/nodes_v%s.csv' % (data_dir, VERSION)
vg_nodes_file='%s/visualgenome-lbl/nodes_v%s.csv' % (data_dir, VERSION)
wn_nodes_file='%s/wordnet/nodes_v%s.csv' % (data_dir, VERSION)
wd_nodes_file='%s/wikidata/nodes_v%s.csv' % (data_dir, VERSION)
fn_nodes_file='%s/framenet/nodes_v%s.csv' % (data_dir, VERSION)
combined_nodes_file='%s/nodes_v%s.csv' % (output_dir, VERSION)
nodes_inputs=[cn_nodes_file,vg_nodes_file,wn_nodes_file,wd_nodes_file, fn_nodes_file]

cn_edges_file='%s/conceptnet/edges_v%s.csv' % (data_dir, VERSION)
vg_edges_file='%s/visualgenome-lbl/edges_v%s.csv' % (data_dir, VERSION)
wn_edges_file='%s/wordnet/edges_v%s.csv' % (data_dir, VERSION)
wd_edges_file='%s/wikidata/edges_v%s.csv' % (data_dir, VERSION)
fn_edges_file='%s/framenet/edges_v%s.csv' %  (data_dir, VERSION)

wn2wn_edges_file='%s/mappings/wn_wn_mappings.csv' % data_dir
wn2wd_edges_file='%s/mappings/wn_wdt_mappings.csv' % data_dir
fn2cn_edges_file='%s/mappings/fn_cn_mappings.csv' % data_dir
combined_edges_file='%s/edges_v%s.csv' % (output_dir, VERSION)
edges_inputs=[cn_edges_file,vg_edges_file,
              wn_edges_file,wd_edges_file,fn_edges_file,
              wn2wn_edges_file,wn2wd_edges_file,fn2cn_edges_file]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

### Combine and store nodes ###

all_dfs=[]
for f in nodes_inputs:
    tmp_df=pd.read_csv(f, sep='\t', converters={5: eval})
    all_dfs.append(tmp_df)

combined_nodes = pd.concat(all_dfs)

#l=combined_nodes[combined_nodes.duplicated(['id'], keep=False)]['id']
#for e in l:
#    if not e.startswith('wn:'): 
#        print(e)

print('combined nodes - number before deduplication:', len(combined_nodes))

# Drop duplicates
combined_nodes.drop_duplicates(subset=['id', 'aliases'], keep = 'first', inplace = True)

#combined_nodes=combined_nodes.groupby(['id'], as_index=False).agg({'aliases': ','.join})

print('combined nodes after deduplication:', len(combined_nodes))
nodes_in_nodes=set(combined_nodes.id.unique())

### Combine and store edges ###

all_dfs=[]
for f in edges_inputs:
    tmp_df=pd.read_csv(f, sep='\t', converters={5: eval})
    all_dfs.append(tmp_df)

combined_edges = pd.concat(all_dfs)

combined_edges['predicate'].replace({"mw:sameAs": "mw:SameAs"}, inplace=True)

print('number of edges before deduplication', len(combined_edges))

# Drop duplicates
combined_edges.drop_duplicates(subset =['subject', 'predicate','object'], 
                                 keep = 'first', inplace = True)

### Analysis and consistency ###

uniq_subjects=combined_edges.subject.unique()
uniq_objects=combined_edges.object.unique()
nodes_in_edges=set(uniq_subjects) | set(uniq_objects)

print('nodes found in edges', len(nodes_in_edges))
print('nodes that have no edges', len(nodes_in_nodes-nodes_in_edges))

missing_nodes=nodes_in_edges-nodes_in_nodes
print('nodes found only in the edges but not in the nodes file', len(missing_nodes))

rows=[]
for m in missing_nodes:
    if m.startswith('wd:'):
        datasource=config.wdt_ds
    elif m.startswith('wn:'):
        datasource=config.wn_ds
    elif m.startswith('/c/en'):
        datasource=config.cn_ds
    a_row=[m, "", "", "", datasource, {}]
    rows.append(a_row)

new_df=pd.DataFrame(rows, columns=config.nodes_cols)
combined_nodes_plus = pd.concat([combined_nodes, new_df])

### Store data ###

combined_nodes_plus.sort_values('id').to_csv(combined_nodes_file, index=False, sep='\t')
combined_edges.sort_values(by=['subject', 'predicate','object']).to_csv(combined_edges_file, index=False, sep='\t')

print('final number of nodes', len(combined_nodes_plus))
print('final number of edges', len(combined_edges))
