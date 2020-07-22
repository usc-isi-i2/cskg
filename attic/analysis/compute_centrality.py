import sys
sys.path.append('../kgtk')
import importlib
import pandas as pd
import numpy as np
import graph_tool.all as gt

import gt.io_utils as gtio
import gt.analysis_utils as gtanalysis

name='wordnet'
datadir='/Users/filipilievski/mcs/cskg/output_v003/%s' % name
mowgli_nodes=f'{datadir}/nodes_v003.csv'
mowgli_edges=f'{datadir}/edges_v003.csv'
output_gml=f'{datadir}/graph.graphml'

#gtio.transform_to_graphtool_format(mowgli_nodes, mowgli_edges, output_gml, True)
g=gtio.load_gt_graph(output_gml.replace(".graphml", '.gt'))

print(g.properties)
print(g.properties[('e', 'predicate')])


print('num nodes', g.num_vertices())
print('num edges', g.num_edges())

print()

g.vp['vertex_betweenness'], g.properties[('e', 'edge_betweenness')] = gtanalysis.compute_betweenness(g)
print('Highest node betweenness')
gtanalysis.get_topn_indices_node(g, 'vertex_betweenness', 5)

#print('Highest edge betweenness')
#gtanalysis.get_topn_indices_edge(g, 'edge_betweenness', 10)

print('Highest edge betweenness')

prop='edge_betweenness'
max_eb=0.0
max_eb_triple=None
for e in g.edges(): 
    bt=g.properties[('e', prop)][e] 
    if bt>max_eb:
        max_eb=bt
        sub=g.vp['_graphml_vertex_id'][tuple(e)[0]]
        pr=g.properties[('e', 'predicate')][e] 
        obj=g.vp['_graphml_vertex_id'][tuple(e)[1]]
        max_eb_triple=(sub, pr, obj)
    
print(max_eb, max_eb_triple)
    
#import IPython
#IPython.embed()
#exit(1)
print()

g.vp['vertex_pagerank'] = gtanalysis.compute_pagerank(g)

max_pr, max_pr_vertex=gtanalysis.get_max_node(g, 'vertex_pagerank')
print('max PR node', max_pr_vertex, max_pr)
print('top N=10 pageranks')
gtanalysis.get_topn_indices_node(g, 'vertex_pagerank', 10)
prs=g.vp['vertex_pagerank'].a

print('max pagerank', np.max(prs))
print('min pagerank', np.min(prs))

pr_data={}
pr_data['PageRank']=list(np.sort(prs))
pr_data['x'] = list(np.arange(len(prs)))

pr_df=pd.DataFrame(pr_data)

print()

hits_eig, g.vp['vertex_hubs'], g.vp['vertex_auth']=gtanalysis.compute_hits(g)
print('HITS eig:', hits_eig)

print('HITS hubs')
gtanalysis.get_topn_indices_node(g, 'vertex_hubs', 10)
print('HITS auth')
gtanalysis.get_topn_indices_node(g, 'vertex_auth', 10)

