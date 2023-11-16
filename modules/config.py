config: dict = {
    'device': 'mps',
    'seq_len': 300,  # 这是最重要的指标之一
    'batch_size': 256,  # batch_size 过小可能导致损失曲线震荡
    'train_days': 50,
    'input_dim': 426,  # = embed_size（自注意力） = feature_size（本题无需embedding）
    'hidden_dim': 64,
    'daily_secs': 3998,
    'dropout': .1,
    'output_dim': 3,
    'num_heads': 2,
    'num_layers': 2,
    'lr': 2 * 1e-4,  # 学习率过大可能导致学习率震荡
    'weight_decay': 5e-3,
    'num_epochs': 10,
    'model_path': './transformer_models/model_round_0',
    'optimizer_path': './transformer_models/optimizer_round_0'
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
