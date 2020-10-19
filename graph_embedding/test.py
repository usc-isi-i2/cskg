
def tsv_process(tsv_file,output_file):
    output = open(output_file,'a')
    with open(tsv_file) as f:
        for line in f:
            content = line.split('\t')[:3]
            if content[1]!='relation': # ignore the first time
                output.write(content[0]+'\t')
                output.write(content[1]+'\t')
                output.write(content[2]+'\n')
    output.close()


from pathlib import Path

if __name__ == "__main__":

    # input_file = 'input/kgtk_atomic.tsv'
    # output_file = 'output/kgtk_atomic'
    # # input_path = Path(input_file)
    
    # # tmp_tsv_path = Path('/tmp') / input_path.name
    

    # # tsv_process(input_path,tmp_tsv_path)

    # input_path = Path(input_file)
    # output_path = Path(output_file)
    # edge_paths = output_path / 'edges_partitioned'
    # checkpoint_path = output_path/'model'
    # print(input_path,edge_paths,checkpoint_path)
  

   

    # tmp_tsv_path = str(Path('/tmp') / input_path.name)
    # print(tmp_tsv_path,type(tmp_tsv_path))
    
    output_file = '1.txt'
