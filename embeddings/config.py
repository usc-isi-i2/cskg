def get_config(entity_path,edge_paths,checkpoint_path,
               entities_structure,relation_structure,dynamic_relations=True, # Graph structure
               dimension=100,global_emb=False,comparator='dot', #  Scoring model
               init_scale=0.1,bias=False,num_epochs=50,num_uniform_negs=1000,loss_fn='softmax',lr=0.1,regularization_coef=1e-3, #Training 
               eval_fraction=0, #Evaluation during training
               num_machines=1,distributed_init_method=None):  # distirbuted mode

    config = dict(
        # I/O data
        entity_path = str(entity_path),
        edge_paths = edge_paths,
        checkpoint_path = checkpoint_path,
        # Graph structure
        entities=entities_structure,
        relations= relation_structure,
        dynamic_relations=dynamic_relations,
        # Scoring model
        dimension=dimension,
        global_emb=global_emb,
        comparator=comparator,
        # Training
        init_scale = init_scale,
        bias=bias,
        num_epochs=num_epochs,
        num_uniform_negs=num_uniform_negs,
        loss_fn=loss_fn,
        lr=lr,
        regularization_coef=regularization_coef,
        # Evaluation during training
        eval_fraction=eval_fraction,  # to reproduce results, we need to use all training data 

        # distribute mode
        num_machines=num_machines, #2
        distributed_init_method=distributed_init_method # 'tcp://128.9.35.196:23456' , # ckg01's ip address

    )

    return config


