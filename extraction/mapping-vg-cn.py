import utils
import config

VERSION=config.VERSION

# INPUT FILE
cn_nodes_file='../output_v%s/conceptnet/nodes_v%s.csv' % (VERSION, VERSION)
vg_nodes_file='../output_v%s/visualgenome/nodes_v%s.csv' % (VERSION, VERSION)

# OUTPUT FILE
edges_file='../output_v%s/mappings/vg_cn_mappings.csv' % VERSION

vg_prefix='vg:'

utils.sameas_to_conceptnet(vg_nodes_file, cn_nodes_file, vg_prefix, edges_file)

