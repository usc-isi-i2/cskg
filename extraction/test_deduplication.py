def combine_dicts(dicts):
    #dicts=x.split(',')
    new_dict=defaultdict(set)
    for d in dicts:
        d=json.loads(d)
        for k in d.keys():
            new_dict[k].add(d[k])
    return new_dict

def merge_and_deduplicate(x):
    return ','.join(list(set(x.split(','))))

def deduplicate_with_transformations(df, join_columns, transformations={'aliases': ','.join, 'pos': ','.join, 'datasource': ','.join, 'label': ','.join, 'other': ','.join}):
    grouped=df.groupby(join_columns, as_index=False).agg(transformations)
    for col in transformations.keys():
        if col not in ['other', 'weight']:
            grouped[col] = grouped[col].apply(merge_and_deduplicate)
        elif col=='other':
            grouped[col] = grouped[col].apply(combine_dicts)

    print(grouped)
    return grouped


x=[[1,'2','bro,sis', 'a,bc,de', '{"a": "b"}', 1.0], [1, '3', 'abc,bro', 'def,de', '{"a": "b"}', 0.35], [2,'3','asd', 'asd,dsa', '{"a": "ad"}', 0.35]]

from collections import defaultdict
import pandas as pd
import json
df=pd.DataFrame(x, columns=['a', 'datasource', 'aliases', 'pos', 'other', 'weight'])

transformations={'aliases': ','.join, 'pos': ','.join, 'datasource': ','.join, 'other': list, 'weight': max}
join_column='a'

df=deduplicate_with_transformations(df, join_column, transformations)
