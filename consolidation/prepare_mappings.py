from kgtk.kgtkformat import KgtkFormat

def print_edge(cols):
    return '\t'.join(cols) + '\n'

def extract(input_file, output_file, source):
    rows=[]
    with open(output_file, 'w') as w:
        columns=['id', 'node1', 'relation', 'node2', 'node1;label', 'node2;label','relation;label', 'relation;dimension', 'source', 'sentence']
        w.write(print_edge(columns))
        with open(input_file, 'r') as f:
            header=next(f)
            for line in f:
                data=line.split('\t')
                data[1]=data[1].replace('same', 'Same')
                if data[1]=='mw:HasInstance':
                    data[1]='fn:HasLexicalUnit'
                id='-'.join(data[:3])
                new_row=[id, *data[:3], "", "", "", "", KgtkFormat.stringify(source), ""]
                w.write(print_edge(new_row))


extract('../input/mappings/wn_wn_mappings.csv', '../tmp/mapping_wn_wn.tsv', "ILI")

extract('../input/mappings/fn_cn_mappings.csv', '../tmp/mapping_fn_cn.tsv', "FNC")

extract('../input/mappings/wn_wdt_mappings.csv', '../tmp/mapping_wn_wd.tsv', "XLN")
