import json
import glob
from collections import defaultdict
import gzip

pick='wikidata'
pick='conceptnet'
pick='framenet'
filename='../tmp/kgtk_%s' % pick

filename='../output/cskg_connected.tsv'

with open("dimensions.json", 'rb') as f:
    dimensions=json.load(f)
    print(len(dimensions.keys()))
    with open(filename, 'r') as myfile:
        dim_counts=defaultdict(int)
        not_covered=set()
        header=next(myfile)

        with gzip.open('../output/cskg_connected_dim.tsv', 'wb') as w:
            w.write(header.encode())
            for line in myfile:
                d=line.split('\t')
                rel_index=2
                if d[rel_index] in dimensions.keys():
                    dim=dimensions[d[rel_index]]
                    dim_counts[dim]+=1
                    d[-3]=dim
                    w.write('\t'.join(d).encode())
                else:
                    not_covered.add(d[rel_index])

        print(dim_counts)
        print(not_covered)
