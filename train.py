from torch.utils.data import DataLoader
from model import RobustModel
from dataset import Dataset
from torch import optim
from torch import nn
import torch.nn.functional as f
import numpy as np
import torch

""" Load Dataset """
batch_size = 120

train_data = Dataset(train=True)
valid_data = Dataset(train=False)

train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size)

""" Create Model """
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = RobustModel()
model = model.to(device)

""" Train & Evaluation """
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

max_correct = 0
valid_accuracy_list = []

for epoch in range(1, 101):
    print(f'{epoch} epoch')
    running_loss = 0.0

    model.train()

    # Train
    for idx, data in enumerate(train_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (idx + 1) % 120 == 0:
            print('[%d, %5d] train loss: %.3f' %
                  (epoch, idx + 1, running_loss / 100))
            running_loss = 0.0

    # Evaluation
    model.eval()
    test_loss = 0
    correct = 0
    total_pred = np.zeros(0)
    total_target = np.zeros(0)

    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            y_hat = model(data)
            test_loss += f.nll_loss(y_hat, target, reduction='sum').item()
            pred = y_hat.argmax(dim=1, keepdim=True)

            total_pred = np.append(total_pred, pred.cpu().numpy())
            total_target = np.append(total_target, target.cpu().numpy())
            correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(valid_data)
        print(f'accuracy : {accuracy}')
        valid_accuracy_list.append(accuracy)

        torch.save(model.state_dict(), f'state/model{epoch}.pt')
        max_correct = correct

print(','.join([str(acc) for acc in valid_accuracy_list]))
print('Finished')
