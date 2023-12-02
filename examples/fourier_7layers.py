import torch
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft

import copy
import random
import time

import torch
import torchvision
import wandb
from rtpt import RTPT



SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

ROOT = '.data'



train_data = datasets.CIFAR10(root=ROOT,
                              train=True,
                              download=True)

means = train_data.data.mean(axis=(0, 1, 2)) / 255
stds = train_data.data.std(axis=(0, 1, 2)) / 255

train_transforms = transforms.Compose([
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(32, padding=2),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=means,
                                                std=stds)
                       ])

test_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=means,
                                                std=stds)
                       ])

train_data = datasets.CIFAR10(ROOT,
                              train=True,
                              download=True,
                              transform=train_transforms)

test_data = datasets.CIFAR10(ROOT,
                             train=False,
                             download=True,
                             transform=test_transforms)

VALID_RATIO = 0.9

n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(train_data,
                                           [n_train_examples, n_valid_examples])

valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms

BATCH_SIZE = 64

train_iterator = data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

valid_iterator = data.DataLoader(valid_data,
                                 batch_size=BATCH_SIZE)

test_iterator = data.DataLoader(test_data,
                                batch_size=BATCH_SIZE)

class Fourier(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.W1 = nn.Parameter(torch.rand(28, 28, 7, 1))
        

        #self.W2 = nn.Parameter(torch.rand(28,28, 8, 7))
        self.W2 = nn.Parameter(torch.rand(28,28, 8, 7))
        self.W3 = nn.Parameter(torch.rand(14,14, 8, 8))
        self.W4 = nn.Parameter(torch.rand(14,14, 8, 8))
        self.W5 = nn.Parameter(torch.rand(7,7, 8, 8))
        self.W6 = nn.Parameter(torch.rand(7,7, 8, 8))
        self.W7 = nn.Parameter(torch.rand(7,7, 10, 8))

        

    def fourier_conv(self, x, W):
        x = torch.fft.fft2(x) #64, 1, 28, 28 ; 64,7,28,28
        
        x = torch.permute(x, (0, 2, 3, 1)).unsqueeze(-1) #64,28,28,1,1 ; 64,28,28,7,1
        
       # 64 , 1 , 28 ,28  -> 28, 28, 7, 1 * 64, 28, 28, 1, 1  7,1 * 1,1 = 7,1. (it considers the last two items and multiply them and gets as a 7,1 vector)
        # 28, 28, 7, 1 will be same for every item in the batch 
        # after every batch we will get a different 28, 28, 7, 1  optimizer.step() will do this 
        x = torch.matmul(torch.complex(W, torch.zeros_like(W)), x) # 64, 28, 28, 7 ,1
        
        x = torch.permute(x.squeeze(-1), (0, 3, 1, 2)) # 64, 7, 28, 28
        
        x = torch.fft.ifft2(x).real # 64, 7, 28, 28
        
        
        # out = np.zeros((x.shape[:-3] + (5,) + x.shape[-2:]))
        # for h in range(H):
        #     for w in range(W):
        #         out[..., :, h, w] = W[h, w]@x[...,:,h,w]
        return x

    def forward(self, x):

        #x = [batch size, 1, 28, 28]
       
        #x = self.W1(x)
        #print(x.shape)
        x = self.fourier_conv(x, self.W1)
        x = F.relu(x)
        
        x = self.fourier_conv(x, self.W2)
        
        x = F.max_pool2d(x, kernel_size=2)
        
        x = self.fourier_conv(x, self.W3)
        x = F.relu(x)
        x = self.fourier_conv(x, self.W4)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.fourier_conv(x, self.W5)
        x = F.relu(x)
        x = self.fourier_conv(x, self.W6)
        x = F.relu(x)
        x = self.fourier_conv(x, self.W7)
        x = F.relu(x)
        x= torch.mean(x,dim=(-2,-1))
        
        m = nn.Softmax( dim = 1)

        x = m(x)
        
       # x = torch.matmul(torch.log(1+torch.abs(x)),self.W1)
        #print(x.shape)

        #x = [batch size, 1, 28, 28]

        
        #print(x.shape)

        #x = [batch size, 1, 14, 14]

        
        #print(x.shape)
        #x = self.W2(x)
        
        # x = torch.matmul(torch.log(1+torch.abs(x)),self.W2)
        #print(x.shape)

        #x = [batch size, 1, 14, 14]

        
        #print(x.shape)

        #x = [batch size, 1, 7, 7]

      
        #print(x.size())

        
        

        #x = [batch size, 1*7*7 = 49]

        h = x

        #x = self.fc_1(x)
        #print(x.shape)

        #x = [batch size, 120]

      
        #print(x.shape)

        #x = self.fc_2(x)
        #print(x.shape)

       # x = [batch size, 84]

        
        #print(x.shape)

        #x = self.fc_3(x)
        #print(x.shape)

       # x = [batch size, output dim]
        #exit()

        return x,h
wandb.init(project="Fourier_7layers_CIFAR10")    
OUTPUT_DIM = 10

model = Fourier(OUTPUT_DIM)
for p in model.parameters():
    nn.init.kaiming_normal_(p.data)
optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

criterion = criterion.to(device)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in tqdm(iterator, desc="Training", leave=False):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

EPOCHS = 100

best_valid_loss = float('inf')
rtpt = RTPT(name_initials='UY', experiment_name='Wavelets', max_iterations=EPOCHS)
rtpt.start()
for epoch in trange(EPOCHS, desc="Epochs"):

    start_time = time.monotonic()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut2-model.pt')

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    wandb.log({"loss":valid_loss, "accuracy":valid_acc})
    rtpt.step()



#for param in Fourier.parameters():
    #print(type(param.data), param.size())

