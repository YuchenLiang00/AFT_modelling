config: dict = {
    'device': 'mps',
    'data_dir': './train_data.npy',
    'label_dir': './train_labels.npy',
    'batch_size': 64,
    'seq_len': 5,
    'train_ratio': .8,
    'hidden_dim': 128,  # = embedding dimension
    'dropout': .1,
    'input_dim': 26,
    'output_dim': 3,
    'num_heads': 2,
    'num_layers': 2,
    'n_steps': 128,
    'pos_enco': True,
    'lr':2 * 1e-5,
    'num_epochs':20,
    'model_path': './transformer_models/label_5'
}