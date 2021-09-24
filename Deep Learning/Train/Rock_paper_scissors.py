from torch import nn
import torch.nn.functional as F
from torchvision import models

class Rock_Paper_Scissors(nn.Module):
    def __init__(self):
        super(Rock_Paper_Scissors,self).__init__()
        self.fc1 = nn.Linear(2048,256)
        self.fc2 = nn.Linear(256,10)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)


        return x 

model = models.resnet50(pretrained = True)

for param in model.parameters():
    param.requires_grad = False

model.fc = Rock_Paper_Scissors()
