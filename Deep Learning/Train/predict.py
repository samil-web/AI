# Imports
import torch
from Rock_paper_scissors import model 
import cv2
from PIL import Image
import os
from torchvision import transforms

model.eval()

def accuracy_test(testloader):
    predictions = []
    correct,total = 0,0
    for i,data in enumerate(testloader,0):
        inputs,labels = data
        outputs = model(inputs)

        _,predicted = torch.max(outputs.data,1)
        predictions.append(outputs)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()
        print('The testing set accuracy of network is %d %%'%(100*correct/total))

def predictor(image,model):
    test_transform = transforms.Compose([transforms.Resize((64,7)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.485,0.456,0.406],
                                            std = [0.229,0.224,0.225])])
    # print(test_transform(image))
    save_path = 'model/Rock_Paper_Scissors.pth'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(save_path,map_location = device))
    model.eval()

    # Generate prediction 
    rps_class  = model(test_transform(image).unsqueeze(0))
    _,predicted = torch.max(rps_class.data,1)

    softmax = torch.nn.Softmax(dim=1)
    ps = softmax(rps_class)
    class_names = {key:val for key,val in enumerate(os.listdir('Dataset/dataset_splits/test'))}
    print(class_names)

    return rps_class,class_names[int(predicted)], ps

test_image = Image.open('test_images/paper1.png')
rgb = test_image.convert('RGB')
print(predictor(rgb, model))