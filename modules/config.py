config: dict = {
    'device': 'mps',
    'seq_len': 100,  # 这是最重要的指标之一 quadratic to model's complexity
    'batch_size': 256,  # batch_size 过小可能导致损失曲线震荡
    'train_days': 50,
    'input_dim': 426,  # = embed_size（自注意力） = feature_size（本题无需embedding）
    'hidden_dim': 64,
    'dropout': .1,
    'output_dim': 3,
    'num_heads': 2,
    'num_layers': 2,
    'lr': 1 * 1e-5,  # 学习率过大可能导致学习率震荡
    'weight_decay': 0,
    'num_epochs': 20,
    'model_path': './model_output/model_round_4',
    'optimizer_path': './model_output/optimizer_round_4'
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
