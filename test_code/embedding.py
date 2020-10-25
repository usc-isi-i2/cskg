import click
from pathlib import Path
import shutil
from config import get_config
import json
import h5py
import os
import torch
from torchbiggraph.config import parse_config
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, setup_logging

# Initializing libiomp5.dylib, but found libomp.dylib already initialized.
# solution: Allow repeat loading dynamic link library 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# method 1
# torch.set_num_threads(2)
# ##Sets the number of threads used for intraop parallelism on CPU. WARNING: 
# # To ensure that the correct number of threads is used, 
# ## set_num_threads must be called before running eager, JIT or autograd code.
OMP_NUM_THREADS=1
os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS



def tsv_process(tsv_file,output_file): 

    output = open(output_file,'w')
    count = 0
    # with open(tsv_file) as f:
    #     for line in f:
    #         content = line.split('\t')[:3]
    #         if content[1]!='relation': # ignore the first time
    #             output.write(content[0]+'\t')
    #             output.write(content[1]+'\t')
    #             output.write(content[2]+'\n')
    #             count+=1
    #         # if count>1000:
            #     break
    with open(tsv_file) as f:
        for line in f:
            content = line.split('\t')[:4]
            if content[1]!='node1': # ignore the first time
                output.write(content[1]+'\t')
                output.write(content[2]+'\t')
                output.write(content[3]+'\n')

    output.close()

@click.command()
@click.option('-i','--input',help='Input KGTK file',required=True, metavar='')
@click.option('-o','--output',help='Output directory', required=True, metavar='')
@click.option('-d','--dimension',help='Dimension of the real space \
	the embedding live in [Default: 100]',default=100, type=int,metavar='')
@click.option('-s','--init_scale',help='Generating the initial \
	embedding with this standard deviation [Default: 0.01]',type=float,default=0.01, metavar='')
@click.option('-c','--comparator',help='Comparator types [Default:dot] Choice: dot | cos | l2 | squared_l2 \
	',default='dot',type=click.Choice(['dot','cos','l2','squared_l2']),metavar='')
@click.option('-b','--bias',help='Whether use the bias choice [Default: False]',type=bool,default=False,metavar='')
@click.option('-e','--num_epochs',help='Training epoch numbers[Default: 100]',type=int,default=100,metavar='')
@click.option('-ge','--global_emb',help='Whether use global embedding [Default: False]',type=bool,default=False,metavar='')
@click.option('-lf','--loss_fn',help='Type of loss function [Default: ranking] \
	Choice: ranking | logistic | softmax ',default='ranking',type=click.Choice(['ranking','logistic','softmax']),metavar='')
@click.option('-lr','--learning_rate',help='Learning rate [Default: 0.1]',type=float,default=0.1,metavar='')
@click.option('-rc','--regularization_coef',help='Regularization coefficient [Default: 1e-3]',type=float,default=1e-3,metavar='')
@click.option('-nn','--num_uniform_negs',help='Negative sampling number [Default: 1000]',type=int,default=1000,metavar='')
@click.option('-dr','--dynamic_relaitons',help='Whether use dynamic relations (when graphs with a \
	large number of relations)[Default: True]',type=bool,default=True,metavar='')
@click.option('-ef','--eval_fraction',help='Fraction of edges withheld from training and used \
	to track evaluation metrics during training [Default: 0.0001]',type=float,default=0.001,metavar='')
@click.option('-nm','--num_machines',help='The number of machines for distributed training [Default: 1]',type=int,default=1,metavar='')
@click.option('-dm','--distributed_init_method',help='A URI defining how to synchronize all \
the workers of a distributed run[Default: None]',default=None,metavar='')
def main(**args):
    """
    Parameters setting and graph embedding
    """
    
    input_path = Path(args['input'])
    output_path = Path(args['output'])

    #prepare  the graph file
    try:  
        tmp_tsv_path = Path('tmp') / input_path.name
        shutil.rmtree(tmp_tsv)
    except:pass
    tsv_process(input_path,tmp_tsv_path)  

    # *********************************************
    # 1. DEFINE CONFIG  
    # *********************************************
    if args['num_machines']>1: # use disrtibuted mode:
        # A good default setting is to set num_machines to half the number of partitions
        num_partitions = args['num_machines']*2
    else:
        num_partitions = 1

    edge_paths = [str(output_path / 'edges_partitioned')]
    checkpoint_path = str(output_path/'model')
    entities= {"all": {"num_partitions": num_partitions}}  #######......
    relations=[  # relation template setting
        {
            "name": "all_edges",
            "lhs": "all",
            "rhs": "all",
            "operator": "complex_diagonal",
        }
    ]

    raw_config = get_config(entity_path=output_path)

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
    if args['num_machines'] == 1: # local
        train(config, subprocess_init=subprocess_init)
    else: # distributed 
        rank = input('Please give the rank of this machine:')
        train(config, subprocess_init=subprocess_init,rank=int(rank))


    # ************************************************
    # 4. EVALUATION  to do ...
    #*************************************************


if __name__ == "__main__":
    main()