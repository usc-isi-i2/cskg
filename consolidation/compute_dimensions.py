import json
import glob
from collections import defaultdict
import gzip

pick='wikidata'
pick='conceptnet'
pick='framenet'
filename='../tmp/kgtk_%s.tsv' % pick

filename='../output/cskg.tsv.gz'

with open("dimensions.json", 'r') as f:
    dimensions=json.load(f)
    print(len(dimensions.keys()))
    with gzip.open(filename, 'rb') as myfile:
        dim_counts=defaultdict(int)
        not_covered=set()
        header=next(myfile).decode()
        with gzip.open('../output/cskg_dim.tsv.gz', 'wb') as w:
            w.write(header.encode())
            for line in myfile:
                d=line.decode().split('\t')
                rel_index=1
                if d[rel_index] in dimensions.keys():
                    dim=dimensions[d[rel_index]]
                    dim_counts[dim]+=1
                    d[-3]=dim
                    w.write('\t'.join(d).encode())
                else:
                    not_covered.add(d[rel_index])

        print(dim_counts)
        print(not_covered)
