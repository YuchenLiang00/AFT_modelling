import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from dataset import LOBDataset
from modules.config import config
from tqdm import tqdm

def cal_f1score(model,valid_iter: DataLoader):
    model.eval()
    y_list = []
    y_hat_list = []
    for X, y in tqdm(valid_iter):
        X = X.to(config['device'])
        with torch.no_grad():
            pred: torch.Tensor = model(X)
            y_hat = pred.argmax(dim=1)
            y_hat_list.append(y_hat.detach())
            y_list.append(y)
    y_hat = torch.concatenate(y_hat_list).to('cpu').numpy()
    y = torch.concatenate(y_list).numpy()
    score = f1_score(y, y_hat, average='macro')
    print(f'macro F1-score = {score:.4f}')
    acc = (y == y_hat).mean()
    print(f'acc={acc:.4%}')
    return score

def test(valid_iter):
    for _,y in valid_iter:
        print(y)
        pass


if __name__ == '__main__':
    model = torch.load('./model_output/model_round_3').to(config['device'])
    config['seq_len']=1
    valid_iter = DataLoader(LOBDataset(is_train=False, config=config, pred_label=0), 
                            batch_size=1024, shuffle=False)
    score = cal_f1score(model, valid_iter)
    # print(score)
    # test(valid_iter)
    