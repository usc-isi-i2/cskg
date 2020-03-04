from kgtk.cskg_utils import collapse_identical_nodes
import pandas as pd
edges_df = pd.read_csv('output_v003/cskg/edges_v003.csv', sep='\t')
nodes_df=pd.read_csv('output_v003/cskg/nodes_v003.csv', sep='\t')

e, n = collapse_identical_nodes(edges_df, nodes_df)

print(len(edges_df), len(e))

print(len(nodes_df), len(n))
