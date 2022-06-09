
def get_default_config():
    """ A default configuration file with information about architecture and HP. """
    config =  {
        'l1_size': 128,
        'l2_size': 64,

        'out_channels_l1': 8,
        'out_channels_l2': 16,        
        'kernel_size_l1': 8,
        'kernel_size_l2': 4,

        'epsilon': 0.1,
        'epsilon_decay_length': 5_000,
        'replay_buffer_size': 256,
        'batch_size': 256,
        'gamma': 0.99,
        'lambda': 0.95,

        'lr':1e-3, # for policy optimizer
        'lr_value_function': 1e-3,

        'tau': 0.97,

        'training_steps': 100_000,
        'training_start': 300,
        'eval_every': 5000,
        'update_every': 1,
        'value_function_learning_repetition': 80,
        
        'training_episode_length': 1000,
        'eval_episode_length': 1000,

        'updates_counter': 0, 
        'target_net_update_freq': 400,
        }
    assert config['batch_size'] < config['training_start']
    return config 