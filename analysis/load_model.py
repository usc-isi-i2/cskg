import h5py
import numpy
import click
import json

@click.command()
@click.option('-o','--output',help='entity embedding output file',required=True, metavar='')
@click.option('-n','--nodes_path',help='entity path',required=True, metavar='')
@click.option('-e','--embeddings_path',help='embeddings path',required=True, metavar='')
def main(**args):
    # DATA_DIR = args['input']  
    OUTPUT_FILE = args['output']   # 1.tsv
    
    # content = h5py.File(input_file,'r')
    # <KeysViewHDF5 ['model', 'optimizer']>  if we load the model.v_xx.h5

    nodes_path =  args['nodes_path']  #'output/kgtk_framenet/entity_names_all_0.json'
    embeddings_path = args['embeddings_path']   #'output/kgtk_framenet/model/embeddings_all_0.v50.h5' 


    with open(nodes_path, 'r') as f:
        node_names = json.load(f)

    with h5py.File(embeddings_path, 'r') as g:
        embeddings = g['embeddings'][:]

    node2embedding = dict(zip(node_names, embeddings))

    f = open(OUTPUT_FILE,'w')
    for key,value in node2embedding.items():
        value = [str(i) for i in value]
        embeddings = ' '.join(value)
        f.write(key+'\t')
        f.write(embeddings+'\n')
    
    f.close()




if __name__ == "__main__":
    main()