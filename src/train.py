from pickletools import optimize
from torch.optim import Adam

from torch.utils.data import DataLoader
from ViT import ViT
import loadutil
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from torchvision.models import resnet18
from tqdm import tqdm
import sys
import os

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def train(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=50):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        file_name = "ViT_" + str(epoch) + ".pth"
        save_path = os.path.join(weight_save_file, file_name)
        torch.save(model.state_dict(), save_path)    
    return model
''' 
def test(validate_dataset, validate_loader, epoch):
    net.eval()
    val_num = len(validate_dataset)
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim = 1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)
    val_accurate = acc / val_num
    file_name = "ViT" + str(epoch) + ".pth"
    save_path = os.path.join(weight_save_file, file_name)
    torch.save(net.state_dict(), save_path)
'''   
    
net = ViT()
loss_function = nn.CrossEntropyLoss()
params = [p for p in net.parameters() if p.requires_grad]
optimizer = Adam(params, lr = 0.0001)
epochs = 50
weight_save_file = "D:\\codefield\\DEEP_LEARNING\\src\\model"

tar_model = train(net, loadutil.dataloaders, loadutil.dataset_sizes, loss_function, optimizer)
# torch.save(tar_model.state_dict(), save_path)
