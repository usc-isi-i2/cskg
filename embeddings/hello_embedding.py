import os
import shutil
import json
import argparse
import h5py
from pathlib import Path
from torchbiggraph.config import parse_config
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, setup_logging

# Allow repeat loading dynamic link library 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main():

    # ==================================================================
    # 0. GET PARAM SETTINGS
    # ==================================================================
    parser = argparse.ArgumentParser(description='Embedding parameters setting')

    parser.add_argument('-dyre','--dr',metavar='',help='Whetherr use dynamic relations',type=bool,default=False)
    parser.add_argument('-gloemb','--gb',metavar='',help='Whether use global embedding',type=bool,default=False)
    parser.add_argument('-epochs','--e', metavar='', help='Traning epoch number',type=int,default=100)
    parser.add_argument('-negsam','--ns', metavar='', help='Number of negatives uniformly sampled from the currently active partition',type=int,default=1000)
    parser.add_argument('-reg_coef','--rc', metavar='', help='Regularization coefficient',type=float,default=0.001)
    parser.add_argument('-eval_fra','--ef', metavar='', help='Fraction of edges withheld from training and \
    used to track evaluation metrics during training.',type=float,default=0.0)
    
    args = parser.parse_args()

    DATA_DIR = args.i
    GRAPH_PATH =  DATA_DIR + '/edges.tsv'
    MODEL_DIR = args.m
    dimension_num = args.d
    dynamic_relations = args.dr
    global_embedding = args.gb
    comparator_type = args.c
    num_epochs = args.e
    num_uniform_negs = args.ns
    loss_function = args.lf
    learning_rate = args.lr
    regularization_coef = args.rc

    try: # delete the files under tmp/model recursively
        shutil.rmtree(MODEL_DIR)
    except:
        pass

    # ==================================================
    # 1. DEFINE CONFIG
    # this dictionary will be used in steps 2. and 3.
    # ==================================================

    raw_config = dict(
        # I/O data
        entity_path=DATA_DIR,
        edge_paths=[
            DATA_DIR + '/edges_partitioned',  # edges info that will be stored
        ],
        checkpoint_path=MODEL_DIR,

        # Graph structure 
        entities={
            "WHATEVER": {"num_partitions": 1}
        },
        relations=[
            {
                "name": "doesnt_matter",
                "lhs": "WHATEVER",
                "rhs": "WHATEVER",
                "operator": "complex_diagonal",
            }
        ],

        # trainging parameters
        dynamic_relations= dynamic_relations,
        dimension = dimension_num,              # silly graph, silly dimensionality
        global_emb=global_embedding,
        comparator=comparator_type,
        num_epochs=num_epochs,
        num_uniform_negs=num_uniform_negs,
        loss_fn=loss_function,
        lr=learning_rate,
        regularization_coef=regularization_coef,
        eval_fraction=eval_fraction,
    )

    # =======================================================
    # 2. TRANSFORM GRAPH TO A PYTORCHBIGGRAPH-FRIENDLY FORMAT
    # =======================================================
    setup_logging()
    config = parse_config(raw_config)
    subprocess_init = SubprocessInitializer()
    input_edge_paths = [Path(GRAPH_PATH)]

    convert_input_data(
        config.entities,
        config.relations,
        config.entity_path,
        config.edge_paths,
        input_edge_paths,
        TSVEdgelistReader(lhs_col=0, rel_col=None, rhs_col=1),
        dynamic_relations=config.dynamic_relations,
    )

    # ===============================================
    # 3. TRAIN THE EMBEDDINGS
    # ===============================================
    train(config, subprocess_init=subprocess_init)

    # =======================================================================
    # 4. LOAD THE EMBEDDINGS
    # output of the process: dict mapping node names to embeddings
    # =======================================================================
    nodes_path = DATA_DIR + '/entity_names_WHATEVER_0.json'
    embeddings_path = MODEL_DIR + "/embeddings_WHATEVER_0.v{NUMBER_OF_EPOCHS}.h5" \
        .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])

    with open(nodes_path, 'r') as f:
        node_names = json.load(f)

    with h5py.File(embeddings_path, 'r') as g:
        embeddings = g['embeddings'][:]

    node2embedding = dict(zip(node_names, embeddings))
    print('embeddings')
    print(node2embedding)


if __name__ == '__main__':
    main()