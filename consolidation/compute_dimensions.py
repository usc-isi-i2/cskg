import json
import glob
from collections import defaultdict

pick='wikidata'
pick='conceptnet'
with open("dimensions.json", 'rb') as f:
    dimensions=json.load(f)
    print(len(dimensions.keys()))
    for graph_file in glob.glob('../tmp/kgtk_*.tsv'):
        dim_counts=defaultdict(int)
        not_covered=set()
        if pick not in graph_file: continue
        print(graph_file)
        with open(graph_file, 'r') as myfile:
            for line in myfile:
                d=line.split('\t')
                if d[1] in dimensions.keys():
                    dim=dimensions[d[1]]
                    dim_counts[dim]+=1
                else:
                    not_covered.add(d[1])

        print(dim_counts)
        print(not_covered)
