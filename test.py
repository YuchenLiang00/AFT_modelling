import torch 

X = torch.randn(10, 3).float().to('mps')
y = torch.tensor([0,1,2,0,0, 1,0,2,1,2]).to('mps')

loss = torch.nn.CrossEntropyLoss()
print(loss(X,y))