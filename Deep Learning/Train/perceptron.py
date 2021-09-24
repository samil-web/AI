# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataloader
import torch.utils.datasets as datasets
import torchvision.transforms as transforms
# We will use MNIST dataset
# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self,input_size,num_classes): #(28*28)-784 nodes
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

model = NN(784,10)#10 values for each of digit- (0,9)numbers
x = torch.randn(64,784)#
print(model(x).shape)

# Set device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
lr_rate = 0.01
batch_size = 64
num_epochs = 1

# Load data
train_dataset = datasets.MNIST(root = 'dataset/',train  =True,transform = transforms.TOTensor(),download = True)
train_loader = Dataloader(dataset = train_dataset,batch_size = batch_size,shuffle = True)
test_dataset = datasets.MNIST(root = 'dataset/',train  =False,transform = transforms.TOTensor(),download = True)
test_loader = Dataloader(dataset = test_dataset,batch_size = batch_size,shuffle = True)

# Initialize network
model = NN(input_size=input_size,num_classes=num_classes).to(device)

# Loss and Optimizer
criterion  = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = lr_rate)

# Train network
for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader):   
        data = data.to(device = device)
        targets = targets.to(device)

        # Get to correct shape
        data = data.reshape(data.shape[0],-1)
        
        # print(data.shape)
        # forward
        scores = model(data)
        loss = criterion(scores,targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam sep
        optimizer.step()
# Check accuracy
def check_accuracy(loader,model):
    num_correct = 0

    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.reshape[0],-1)

            scores = model(x)
            # 64*10
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            print(f'Got{num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 

        model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)