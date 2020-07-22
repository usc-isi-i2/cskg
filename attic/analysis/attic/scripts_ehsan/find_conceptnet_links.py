from graph_tool.all import Graph, load_graph_from_csv, all_paths, find_vertex
import pandas as pd
import os
import numpy as np
from loguru import logger


from itertools import cycle
from itertools import islice
def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


conceptnet_path = os.path.expanduser('~/project/KB_dump/conceptnet/conceptnet-en.csv')

g = load_graph_from_csv(conceptnet_path, directed=False, 
                        eprop_types=['string', 'string'], 
                        string_vals=True)

prefix = '/c/en/'
entities = [
    ['capoeira', 'hand', 'cartwheel', 'shirt', 'handstand'],
    ['sunscreen', 'skateboarding', 'soccer', 'tan', 'rubbing'],
    ['cream', 'mascara', 'writing', 'lifting', 'dictaphone'],
]


blackListVertex = set([find_vertex(g, prop=g.properties[('v', 'name')],match=prefix+b)[0] 
                       for b in ['object', 'thing']])

blackListEdge = set(['/r/DerivedFrom', '/r/RelatedTo'])
print('#'*20)
for elist in entities:
    print(elist)
    qid = find_vertex(g, prop=g.properties[('v', 'name')],match=prefix+elist[0])[0]
    for a in elist[1:]:
        aid = find_vertex(g, prop=g.properties[('v', 'name')],match=prefix+a)[0]
        for vp, ep in zip(all_paths(g, source=qid, target=aid, cutoff=4),
                          all_paths(g, source=qid, target=aid, cutoff=4, edges=True)):
            
            if len(set(vp).intersection(blackListVertex)) > 0:
                continue
            
            
            rrout = []
            for i in range(len(vp)):
                rrout.append(g.properties[('v', 'name')][vp[i]])
                if i == (len(vp)-1):
                    break
                rrout.append(g.properties[('e', 'c0')][ep[i]])
            
            if len(set(rrout).intersection(blackListEdge)) > 0:
                continue
            
            print(rrout)
            
    print('#'*20)
# all_paths(g, source='/c/en/orange', target='/c/en/food', cutoff=2)

# find_vertex(g, prop=g.properties[('v', 'name')],match='/c/en/orange')

# paths = [i for i in all_paths(g, source=1468, target=2532, cutoff=4)]
# print('#'*20)