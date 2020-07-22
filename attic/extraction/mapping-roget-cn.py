import utils
import config

VERSION=config.VERSION

# INPUT FILE
cn_nodes_file='../output_v%s/conceptnet/nodes_v%s.csv' % (VERSION, VERSION)
roget_nodes_file='../output_v%s/roget/nodes_v%s.csv' % (VERSION, VERSION)

# OUTPUT FILE
edges_file='../output_v%s/mappings/rg_cn_mappings.csv' % VERSION

roget_prefix='rg:'

utils.sameas_to_conceptnet(roget_nodes_file, cn_nodes_file, roget_prefix, edges_file)

