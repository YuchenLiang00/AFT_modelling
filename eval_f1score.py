import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from dataset import LOBDataset
from modules.config import config
from tqdm import tqdm

def cal_f1score(model,valid_iter: DataLoader):
    model.eval()
    y_list = []
    pred_list = []
    for X, y in tqdm(valid_iter):
        X, y = X.to(config['device']), y.to(config['device'])
        with torch.no_grad():
            pred: torch.Tensor = model(X)
            pred_list.append(pred)
            y_list.append(y)
    pred = torch.concatenate(pred_list,)
    _, y_hat = torch.max(pred, dim=1)
    y_hat = y_hat.to('cpu').numpy()
    y = torch.concatenate(y_list).to('cpu').numpy()
    score = f1_score(y, y_hat,average='macro')

    return score

def test(valid_iter):
    for _,y in valid_iter:
        print(y)
        pass


if __name__ == '__main__':
    model = torch.load('./transformer_models/model_round_2').to(config['device'])
    valid_iter = DataLoader(LOBDataset(is_train=True, config=config, pred_label=0),batch_size=32,shuffle=False)
    score = cal_f1score(model, valid_iter)
    print(score)
    # test(valid_iter)
    