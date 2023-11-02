config: dict = {
    'device': 'mps',
    'data_dir': './data/train_data.npy',
    'label_dir': './data/train_labels.npy',
    'batch_size': 10,  # 在本题中要动态变化
    'seq_len': 600,  # 这是最重要的指标之一
    'train_days':48,
    'hidden_dim': 22,  # = embed_size（自注意力） = feature_size（本题无需embedding）
    'dropout': .1,
    'output_dim': 3,
    'num_heads': 2,
    'num_layers': 2,
    'pos_enco': True,
    'lr': 2 * 1e-5,
    'num_epochs': 10,
    'model_path': './transformer_models/label_5',
    'daily_secs':3998,
    
}

'''
注意到
为了保持encoder及decoder的层可以堆叠
需要保证每个层的输入和输出的维度一致
因此
需要保证 embed_size = hidden_size
'''