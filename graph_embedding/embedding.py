import argparse
from pathlib import Path
import shutil
from config import get_config
import json
import h5py
import os
from torchbiggraph.config import parse_config
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, setup_logging


# Initializing libiomp5.dylib, but found libomp.dylib already initialized.
# solution: Allow repeat loading dynamic link library 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def tsv_process(tsv_file,output_file):
    # if already exists, then delete
    if Path(output_file).exists:
        Path(output_file).unlink()

    output = open(output_file,'a')
    count = 0
    with open(tsv_file) as f:
        for line in f:
            content = line.split('\t')[:3]
            if content[1]!='relation': # ignore the first time
                output.write(content[0]+'\t')
                output.write(content[1]+'\t')
                output.write(content[2]+'\n')
                count+=1
            if count>200000:
                break
    output.close()


def main():

    # *********************************************
    # 0. GET PARAM SETTINGS
    # *********************************************
    parser = argparse.ArgumentParser(description='Embedding parameters setting', add_help=False)
    req = parser.add_argument_group('required arguments')
    req.add_argument('-i','--input', action='store', dest='input', help='Input KGTK file',required=True, metavar='')
    req.add_argument('-o','--output', action='store', dest='output', help='Output embedding directory', required=True, metavar='')
    uni = parser.add_argument_group('optional arguments')
    uni.add_argument('-h ', '--help', action='help', help='Show help message and exit')
    uni.add_argument('-d','--dimension', action='store', dest='dimiension', help='Dimension of the real space the embedding live in [Default: 10]', type=int,default=10, metavar='')
    uni.add_argument('-s','--init_scale', action='store', dest='init_scale', help='Generating the initial embedding with this standard deviation [Default: 0.01]',type=float,default=0.01, metavar='')
    uni.add_argument('-c','--comparator', action='store', dest='comparator',help='Comparator types [Default: dot]', default='dot',choices=['dot','cos','l2','squared_l2'],metavar='')
    uni.add_argument('-b','--bias', action='store', dest='bias', help='Whether use the bias choice [Default: False]',  type=bool,default=False,metavar='')
    uni.add_argument('-lf','--loss_fn', action='store', dest='loss_fn', help='Type of loss function [Default: logistic]',default='logistic',choices=['ranking','logistic','softmax'],metavar='')
    uni.add_argument('-lr','--learn_rate', action='store', dest='lr',help='Learning rate [Default: 0.1]',type=float,default=0.1,metavar='')
    args = parser.parse_args()

    
    input_path = Path(args.input)
    output_path = Path(args.output)

    #prepare  the graph file
    try:  
        tmp_tsv_path = Path('/tmp') / input_path.name
        shutil.rmtree(tmp_tsv)
    except:pass
    tsv_process(input_path,tmp_tsv_path)  

    # *********************************************
    # 1. DEFINE CONFIG
    # *********************************************
    edge_paths = [str(output_path / 'edges_partitioned')]
    checkpoint_path = str(output_path/'model')
    entities= {"all": {"num_partitions": 1}}  #######......
    relations=[
        {
            "name": "all_edges",
            "lhs": "all",
            "rhs": "all",
            "operator": "complex_diagonal",
        }
    ]

    raw_config = get_config(entity_path=output_path,edge_paths=edge_paths,checkpoint_path=checkpoint_path,
        entities_structure=entities,relation_structure=relations,dynamic_relations=True,
        dimension=args.dimiension,global_emb=False,comparator=args.comparator,              
        init_scale=args.init_scale,bias=args.bias,num_epochs=50,num_uniform_negs=1000,loss_fn=args.loss_fn,lr=args.lr,regularization_coef=1e-3,
        eval_fraction=0)

    # **************************************************
    # 2. TRANSFORM GRAPH TO A BIGGRAPH-FRIENDLY FORMAT
    # **************************************************
    setup_logging()
    config = parse_config(raw_config)
    subprocess_init = SubprocessInitializer()
    input_edge_paths = [tmp_tsv_path] 

    convert_input_data(
        config.entities,
        config.relations,
        config.entity_path,
        config.edge_paths,
        input_edge_paths,
        TSVEdgelistReader(lhs_col=0, rel_col=1, rhs_col=2),
        dynamic_relations=config.dynamic_relations,
    )

    # ************************************************
    # 3. TRAIN THE EMBEDDINGS
    #*************************************************
    train(config, subprocess_init=subprocess_init)

if __name__ == '__main__':
    main()
    