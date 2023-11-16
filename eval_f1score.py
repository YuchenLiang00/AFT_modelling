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
    print(score)
    acc = (y == y_hat).mean()
    print(acc)
    return score

def test(valid_iter):
    for _,y in valid_iter:
        print(y)
        pass


if __name__ == '__main__':
    model = torch.load('./transformer_models/model_round_4').to(config['device'])
    valid_iter = DataLoader(LOBDataset(is_train=False, config=config, pred_label=0, require_stride=False), 
                            batch_size=32, shuffle=False)
    score = cal_f1score(model, valid_iter)
    # print(score)
    # test(valid_iter)
    