import utils
import config

VERSION=config.VERSION

# INPUT FILE
cn_nodes_file='../output_v%s/conceptnet/nodes_v%s.csv' % (VERSION, VERSION)
at_nodes_file='../output_v%s/atomic/nodes_v%s.csv' % (VERSION, VERSION)

# OUTPUT FILE
edges_file='../output_v%s/mappings/at_cn_mappings.csv' % VERSION

at_prefix='at:'

utils.sameas_to_conceptnet(at_nodes_file, cn_nodes_file, at_prefix, edges_file)

