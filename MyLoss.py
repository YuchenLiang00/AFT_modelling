import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MultiClassFocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2, alpha=None):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # inputs = [batch_size, 1, output_size]
        # targets = [batch_size, 1, 1]
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        # 这里不能直接使用crossEntropyLoss，因为需要对每条样本加权，所以要求出每条样本的各类别概率
        # [batch_size, output_size]
        ce_loss = F.binary_cross_entropy(probs, targets_one_hot, reduction='none')
        p_t = torch.exp(-ce_loss)
        modulating_factor = (1 - p_t) ** self.gamma  # 概率越大（预测越准），对Loss贡献越小

        # Apply class weights if alpha is provided
        if self.alpha is not None:
            modulating_factor = modulating_factor \
            * self.alpha[targets].reshape(modulating_factor.shape[0], 1)

        focal_loss = (ce_loss * modulating_factor).mean(dim=1)
        return focal_loss.mean()


if  __name__ == '__main__':
    # Dummy data for illustration
    num_classes = 5
    inputs = torch.randn((32, num_classes))  # Batch size 32, number of classes 5
    targets = torch.randint(0, num_classes, (32,))  # Random class labels

    # Instantiate the model and the multi-class focal loss
    model = nn.Linear(num_classes, num_classes)  # A simple linear model for illustration
    multi_class_criterion = MultiClassFocalLoss(5, gamma=2, alpha=None)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)
        
        # Compute multi-class focal loss
        loss = multi_class_criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss for monitoring training progress
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
