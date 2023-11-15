config: dict = {
    'device': 'mps',
    'seq_len': 300,  # 这是最重要的指标之一
    'stride': 10,
    'batch_size': 32,
    'train_days': 50,
    'hidden_dim': 426,  # = embed_size（自注意力） = feature_size（本题无需embedding）
    'daily_secs': 3998,
    'dropout': .1,
    'output_dim': 3,
    'num_heads': 2,
    'num_layers': 2,
    'pos_enco': True,
    'lr': 2 * 1e-5,
    'num_epochs': 20,
    'model_path': './transformer_models/model_round_4',
    'optimizer_path': './transformer_models/optimizer_round_4'
}

'''
注意到
为了保持encoder及decoder的层可以堆叠
需要保证每个层的输入和输出的维度一致
因此
需要保证 embed_size = hidden_size
***
hidden_num必须整除num_heads
'''
