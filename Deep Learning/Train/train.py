# Imports
import time
import os
import copy
import numpy as np
import torch 
from torch import nn
from torch import optim
from torch.autograd import Variable
from data_handler import dataset_loader
from Rock_paper_scissors import model
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary

# Training function
def train_model(model,criterion,optimizer,dataloaders,device,num_epochs = 25):
    since = time.time()
    
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch+1,num_epochs))
        print('-'*10)

        try:
            for phase in ['train','val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0.0

                for inputs,labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase =='train'):
                        outputs = model(inputs)
                        _,preds = torch.max(outputs,1)
                        loss = criterion(outputs,labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()*inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss/dataset_sizes[phase]
                    epoch_acc = running_corrects/dataset_sizes[phase]

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))

                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model = copy.deepcopy(model.state_dict())


                print()

        except Exception as e:
            with open('log.txt','a') as file:
                file.write(str(e))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model)

    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_path = "Dataset/dataset_splits/train"
val_path = "Dataset/dataset_splits/val"
test_path = "Dataset/dataset_splits/test"

train_loader, val_loader, _, train_size, val_size, _ = dataset_loader(train_path, val_path, test_path)
# train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)
dataloaders = {'train': train_loader, 'val': val_loader}

dataset_sizes = {'train': train_size, 'val': val_size}

model = model.to(device)
print(summary(model, (3, 224, 224)))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

model = train_model(model, criterion, optimizer, dataloaders, device, num_epochs=20)
torch.save(model.state_dict(), 'model/Rock_Paper_Scissors.pth')
