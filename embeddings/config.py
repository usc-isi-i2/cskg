def get_config(**args): 
    
    num_part = args['num_machines'] if args['num_machines']==1 else args['num_machines']*2

    config = dict(

        # I/O data
        entity_path = args['output'],
        edge_paths = [ args['output']+'/edges_partitioned'],
        checkpoint_path = args['output'] + '/model',

        # Graph structure
        entities= {"all": {"num_partitions": num_part }}  ,
        relations=[  # relation template setting
        {
            "name": "all_edges",
            "lhs": "all",
            "rhs": "all",
            "operator": "complex_diagonal",
        }
        ],
        dynamic_relations=args['dynamic_relaitons'],

        # Scoring model
        dimension=args['dimension'],
        global_emb=args['global_emb'],
        comparator=args['comparator'],

        # Training
        init_scale = args['init_scale'],
        bias=args['bias'],
        num_epochs=args['num_epochs'],
        num_uniform_negs=args['num_uniform_negs'],
        loss_fn=args['loss_fn'],
        lr=args['learning_rate'],
        regularization_coef=args['regularization_coef'],

        # Evaluation during training
        eval_fraction=args['eval_fraction'],  # to reproduce results, we need to use all training data 

        # distribute mode
        num_machines=args['num_machines'],
        distributed_init_method=args['distributed_init_method'] # 'tcp://128.9.35.196:23456' , # ckg01's ip address

    )

    return config


