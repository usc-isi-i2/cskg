from pathlib import Path
import argparse

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




def main():

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
    
    # output_file = '1.txt'

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
    uni.add_argument('-e','--epoch_num', action='store', dest='num_epochs',help='Training epoch numbers[Default: 50]',type=int,default=50,metavar='')
    uni.add_argument('-lf','--loss_fn', action='store', dest='loss_fn', help='Type of loss function [Default: logistic]',default='logistic',choices=['ranking','logistic','softmax'],metavar='')
    uni.add_argument('-lr','--learn_rate', action='store', dest='lr',help='Learning rate [Default: 0.1]',type=float,default=0.1,metavar='')
    uni.add_argument('-dr','--dy_relaiton',action='store',dest='dr',help='Whether use dynamic relations (when graphs with a large number of relations)[Default: True]',type=bool,default=True,metavar='')
    uni.add_argument('-ef','--eval_frac',action='store',dest='eval_fraction',help='Fraction of edges withheld from training and used to track evaluation metrics during training.[Default: 0.0]',type=float,default=.0,metavar='')

    args = parser.parse_args()


if __name__ == "__main__":
    main()