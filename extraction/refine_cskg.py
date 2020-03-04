import kgtk.gt.io_utils as gtio
import kgtk.gt.topology_utils as gtt

name='cskg'
datadir='/Users/filipilievski/mcs/cskg/output_v003/%s' % name
mowgli_nodes=f'{datadir}/nodes_v003.csv'
mowgli_edges=f'{datadir}/edges_v003.csv'
output_gml=f'{datadir}/graph.graphml'

gtio.transform_to_graphtool_format(mowgli_nodes, mowgli_edges, output_gml, True)
g=gtio.load_gt_graph(output_gml.replace(".graphml", '.gt'))

g_well_connected=gtt.get_nodes_with_degree(g, 2, 1000000)

print(g.num_vertices(), g.num_edges())
print(g_well_connected.num_vertices(), g_well_connected.num_edges())
