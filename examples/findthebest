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
from tqdm.notebook import tqdm, trange
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

class Wavelets(nn.Module):
    def __init__(self, output_dim,hidden_dim):
        super().__init__()


        self.Wl00 = nn.Parameter(torch.randn(3,hidden_dim))
        self.Wh00 = nn.Parameter(torch.randn(3, hidden_dim))
        self.Wh01 = nn.Parameter(torch.randn(3, hidden_dim))
        self.Wh02 = nn.Parameter(torch.randn(3, hidden_dim))

        self.Wl10 = nn.Parameter(torch.randn(hidden_dim,hidden_dim))
        self.Wh10 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Wh11 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Wh12 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.Wl20 = nn.Parameter(torch.randn(hidden_dim,hidden_dim))
        self.Wh20 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Wh21 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Wh22 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.Wl30 = nn.Parameter(torch.randn(hidden_dim,hidden_dim))
        self.Wh30 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Wh31 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Wh32 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.Wl40 = nn.Parameter(torch.randn(hidden_dim,hidden_dim))
        self.Wh40 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Wh41 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.Wh42 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.Wl50 = nn.Parameter(torch.randn(hidden_dim,10))
        self.Wh50 = nn.Parameter(torch.randn(hidden_dim, 10))
        self.Wh51 = nn.Parameter(torch.randn(hidden_dim, 10))
        self.Wh52 = nn.Parameter(torch.randn(hidden_dim, 10))

        # self.Wl60 = nn.Parameter(torch.randn(hidden_dim,hidden_dim))
        # self.Wh60 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        # self.Wh61 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        # self.Wh62 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.Wl60 = nn.Parameter(torch.randn(10,10))
        self.Wh60 = nn.Parameter(torch.randn(10, 10))
        self.Wh61 = nn.Parameter(torch.randn(10, 10))
        self.Wh62 = nn.Parameter(torch.randn(10, 10))


        #self.fc_1 = nn.Linear(640, 10)





    def forward_wavelet_transformation(self,x):
        xfm = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b').to(device)
        return xfm(x)



    def Yl_transform(self,Yl_, W):


        Yl_ = torch.matmul(torch.permute(Yl_, (0,2,3,1)), W)
        Yl_ = torch.permute(Yl_, (0,3,1,2))
        return Yl_


    def Yh_transform(self,a, W):


        a = torch.permute(a, (0,2,3,4,5,1))
        a = torch.matmul(a, W)
        a = torch.permute(a, (0,5,1,2,3,4))
        return a



    def inverse_wavelet_transformation(self,Yl_,Yh_0,Yh_1,Yh_2):
        ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').to(device)
        x = ifm((Yl_, [Yh_0, Yh_1, Yh_2]))
        return x

        # self.fc_1 = nn.Linear(16 * 4 * 4, 120)
        # self.fc_2 = nn.Linear(120, 84)
        # self.fc_3 = nn.Linear(84, output_dim)


    def forward(self, x):



        Yl,Yh = self.forward_wavelet_transformation(x)


        Yl = self.Yl_transform(Yl,self.Wl00)

        Yh[0] = self.Yh_transform(Yh[0], self.Wh00)

        Yh[1] = self.Yh_transform(Yh[1], self.Wh01)
        Yh[2] = self.Yh_transform(Yh[2], self.Wh02)
        x = self.inverse_wavelet_transformation(Yl,Yh[0],Yh[1],Yh[2])







        x = F.relu(x)


        Yl,Yh = self.forward_wavelet_transformation(x)

        Yl = self.Yl_transform(Yl,self.Wl10)

        Yh[0] = self.Yh_transform(Yh[0], self.Wh10)

        Yh[1] = self.Yh_transform(Yh[1], self.Wh11)
        Yh[2] = self.Yh_transform(Yh[2], self.Wh12)
        x = self.inverse_wavelet_transformation(Yl,Yh[0],Yh[1],Yh[2])
        x = F.relu(x)


        x = F.max_pool2d(x, kernel_size=2)

        Yl,Yh = self.forward_wavelet_transformation(x)

        Yl = self.Yl_transform(Yl,self.Wl20)

        Yh[0] = self.Yh_transform(Yh[0], self.Wh20)

        Yh[1] = self.Yh_transform(Yh[1], self.Wh21)
        Yh[2] = self.Yh_transform(Yh[2], self.Wh22)
        x = self.inverse_wavelet_transformation(Yl,Yh[0],Yh[1],Yh[2])

        x = F.relu(x)

        #x = x.view(x.shape[0], -1)
        Yl,Yh = self.forward_wavelet_transformation(x)

        Yl = self.Yl_transform(Yl,self.Wl30)

        Yh[0] = self.Yh_transform(Yh[0], self.Wh30)

        Yh[1] = self.Yh_transform(Yh[1], self.Wh31)
        Yh[2] = self.Yh_transform(Yh[2], self.Wh32)
        x = self.inverse_wavelet_transformation(Yl,Yh[0],Yh[1],Yh[2])
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        Yl,Yh = self.forward_wavelet_transformation(x)

        Yl = self.Yl_transform(Yl,self.Wl40)

        Yh[0] = self.Yh_transform(Yh[0], self.Wh40)

        Yh[1] = self.Yh_transform(Yh[1], self.Wh41)
        Yh[2] = self.Yh_transform(Yh[2], self.Wh42)
        x = self.inverse_wavelet_transformation(Yl,Yh[0],Yh[1],Yh[2])
        x = F.relu(x)


        Yl,Yh = self.forward_wavelet_transformation(x)

        Yl = self.Yl_transform(Yl,self.Wl50)

        Yh[0] = self.Yh_transform(Yh[0], self.Wh50)

        Yh[1] = self.Yh_transform(Yh[1], self.Wh51)
        Yh[2] = self.Yh_transform(Yh[2], self.Wh52)
        x = self.inverse_wavelet_transformation(Yl,Yh[0],Yh[1],Yh[2])

        x = F.relu(x)


        Yl,Yh = self.forward_wavelet_transformation(x)

        Yl = self.Yl_transform(Yl,self.Wl60)

        Yh[0] = self.Yh_transform(Yh[0], self.Wh60)

        Yh[1] = self.Yh_transform(Yh[1], self.Wh61)
        Yh[2] = self.Yh_transform(Yh[2], self.Wh62)
        x = self.inverse_wavelet_transformation(Yl,Yh[0],Yh[1],Yh[2])
        x = F.relu(x)
        x= torch.mean(x,dim=(-2,-1))
        # #x = F.relu(x)
        # print(x.shape)
        # exit()
        # #x = torch.mean(x)
        # #print(x.shape)
        # #exit()

        #x = x.view(x.shape[0], -1)


        # #x = torch.mean(x)

        # x = self.fc_1(x)
        
        m = nn.Softmax( dim = 1)

        x = m(x)








        h = x







        return x, h

wandb.init(project="dtcwt_CIFAR10", config=dict(hidden_dim=32))

OUTPUT_DIM = 10

model = Wavelets(OUTPUT_DIM, hidden_dim=wandb.config.hidden_dim)
#model = Wavelets(OUTPUT_DIM, 8)

optimizer = optim.Adam(model.parameters(), lr= 0.01)

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

EPOCHS = 20

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
