import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import kgtk.gt.analysis_utils as gtanalysis   
import kgtk.gt.io_utils as gtio

for name in ['conceptnet', 'visualgenome', 'wikidata', 'wordnet', 'framenet', 'cskg', 'cskg_merged']:
#for name in ['cskg_merged']:
    print(name)
    
    datadir='/Users/filipilievski/mcs/cskg/output_v003/%s' % name

    mowgli_nodes=f'{datadir}/nodes_v003.csv'
    mowgli_edges=f'{datadir}/edges_v003.csv'
    output_gml=f'{datadir}/graph.graphml'

    plottype='loglog'
    base=10
    xlabel='degree'
    ylabel='# nodes'
    directions=['in', 'out', 'total']

    try:
        #gtio.transform_to_graphtool_format(mowgli_nodes, mowgli_edges, output_gml, True)
        g=gtio.load_gt_graph(output_gml.replace(".graphml", '.gt'))
    except FileNotFoundError:
        gtio.transform_to_graphtool_format(mowgli_nodes, mowgli_edges, output_gml, True)
        g=gtio.load_gt_graph(output_gml.replace(".graphml", '.gt'))
        

    print(gtanalysis.get_topN_relations(g))

    degrees={}
    for direction in directions:
        stats=gtanalysis.compute_stats(g, direction)
        print(stats)


        degree_data=gtanalysis.compute_node_degree_hist(g, direction)
        degrees[direction]=list(degree_data[0])

    print('max degree', len(degrees['total'])-1)

    len_degrees=len(degrees['total'])
    for d in directions:
        add_zeros=[0]*(len_degrees-len(degrees[d]))
        degrees[d] = degrees[d] + add_zeros
    #    print(add_zeros)

    degrees['x'] = list(np.arange(len(degrees['total'])))

    df = pd.DataFrame(degrees)

    sns.set(rc={"font.size":20,"axes.titlesize":22,"axes.labelsize":15, 'xtick.labelsize': 15, 'ytick.labelsize': 15}, style="whitegrid")

    f, axes = plt.subplots(1,3, figsize=(24, 6), sharey=True)#, fontsize=20,fontweight='bold')
    for i, d in enumerate(directions):
        p=sns.lineplot(data=df, x='x', y=d, ax=axes[i])
        p.set(xscale="log", yscale="log", title=d + ' degree', xlabel='degree', ylabel='frequency')

    f.savefig(f'../img/degrees_{name}.png')


    g.vp['vertex_pagerank'] = gtanalysis.compute_pagerank(g)

    max_pr, max_pr_vertex=gtanalysis.get_max_node(g, 'vertex_pagerank')

    print('MAX PR', max_pr_vertex, max_pr)

    print('Max pageranks')
    gtanalysis.get_topn_indices(g, 'vertex_pagerank', 5)

    prs=g.vp['vertex_pagerank'].a
    print('pagerank max', np.max(prs))
    print('pagerank min', np.min(prs))

    pr_data={}
    pr_data['PageRank']=list(np.sort(prs))
    pr_data['x'] = list(np.arange(len(prs)))

    pr_df=pd.DataFrame(pr_data)

    fig = plt.figure(figsize=(8,6))
    plt.loglog(np.flip(np.sort(prs)))#, basey=base)
    plt.ylabel('PageRank')
    plt.xlabel('nodes')
    plt.title('PageRank value distribution')

    fig.savefig(f'../img/pagerank_{name}.png')
