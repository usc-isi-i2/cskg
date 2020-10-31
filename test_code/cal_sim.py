import click 
from annoy import AnnoyIndex

"""
Use annoy to get k-nearnest nerighbors of certain entity
"""

# get certain entity's top k neighbors based on cos sim
def topk_entity(annoy_index,entity_dict,entity,k):

    topk=[]
    ent_index = None
    ent_name = None
    if type(entity) == str: # use entity name as search key  => 'wd:Q419890'
        ent_index = entity_dict[entity]
        ent_name = entity
    elif  type(entity) == int: # use entity index as search key =>1
        ent_index = entity
        ent_name = entity_dict[ent_index]
        
    # get topk similar entites
    indices = annoy_index.get_nns_by_item(ent_index,k)
    for i in indices:
        distance = annoy_index.get_distance(ent_index,i)
        cos_sim = 1- distance
        print(f'{entity_dict[i]}:,{cos_sim}')
        topk.append((entity_dict[i],cos_sim))

    return topk

@click.command()
@click.option('-i','--input',help='Entities embedding file',required=True, metavar='')
@click.option('-d','--dimension',help='Vector dimensions of an entity',default=100, metavar='')
# @click.option('-a','--annoy_index',help='Annoy index for embedding',default=None)
# @click.option('-e','--entity_dict',help='Entity name dictionary',default={})
def main(**args):

    input_file = args['input']
    entity_dict = {}
    dimension = args['dimension'] 
    annoy_index = AnnoyIndex(dimension, 'angular') 

    with open(input_file, 'r') as f:
        for index,line in enumerate(f):
            line = line.split('\t')
            entity_name = line[0]
            entity_vec =  [ float(i) for i in line[1:]]
            entity_dict[entity_name] = index
            entity_dict[index] = entity_name
            annoy_index.add_item(index, entity_vec)
                 
    annoy_index.build(50) # 50  binary trees
    annoy_index.save('test.ann')


    ## test : get topk similar entites  example: entity index:1 k:10 
    topk_entity(annoy_index,entity_dict,1,10)

    



if __name__ == "__main__":
    ## prepare the data and get the annoy index
    annoyIndex,entity_dict = main()





