import h5py
import numpy
import click
import json
from torchbiggraph.config import parse_config
from torchbiggraph.converters.export_to_tsv import *

@click.command()
@click.option('-i','--input',help='Embedding folder after training',required=True, metavar='')
@click.option('-eo','--entities_output',help='Entities output tsv file',required=True,metavar='')
@click.option('-ro','--relation_types_output',help='Relation types output tsv file',required=True, metavar='')
def main(**args):
    
    DATA_DIR = args['input']
    # nodes_path =  args['nodes_path']  #'output/kgtk_framenet/entity_names_all_0.json'
    # embeddings_path = args['embeddings_path']   #'output/kgtk_framenet/model/embeddings_all_0.v50.h5' 
    entities_output =args['entities_output']
    relation_types_output = args['relation_types_output']

    config_dir = DATA_DIR + '/model/config.json'
    print(config_dir)

    f = open(config_dir)
    config_dict = json.load(f)
    f.close()

    config = parse_config(config_dict)

    with open(entities_output, "xt") as entities_tf, open(
        relation_types_output, "xt"
    ) as relation_types_tf:
        make_tsv(config, entities_tf, relation_types_tf)


if __name__ == "__main__":
    main()