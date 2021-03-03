# import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy
from sklearn.metrics import accuracy_score,f1_score,roc_curve,precision_recall_curve,average_precision_score,auc
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,matthews_corrcoef,roc_auc_score
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("import finished")

program_name = "20-batch32-lr0.01-SGD"
'''
# unzip datafile
import zipfile

f = zipfile.ZipFile("./q1_data.zip", 'r')  

for file in f.namelist():

    f.extract(file, "./")  

f.close()
'''

# Data preprocessing
# DEFINE
BATCH_SIZE = 32
num_epochs = 300
start_epoch = 0
save_path = "./models" + program_name + ".pth"
CLASS_COARSE = 20
CLASS_FINE = 100
LR = 0.01

print(torch.cuda.is_available())
print(torch.cuda.current_device())

# Redefine train dataset class
class Train_Dataset(Dataset):
    def __init__(self, img_train, img_label, transform=None, target_transform=None):
        self.imgs = img_train
        self.labels = img_label[:, 1]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_0 = self.imgs[index][0]
        img_1 = self.imgs[index][1]
        img_2 = self.imgs[index][2]
        label = self.labels[index]
        img = np.array([img_0, img_1, img_2])
        img = img.transpose(1,2,0)
        img = Image.fromarray(np.uint8(img))
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

# define valid dataset class
class Valid_Dataset(Dataset):
    def __init__(self, img_train, img_label, transform=None, target_transform=None):
        self.imgs = img_train
        self.labels = img_label[:, 1]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_0 = self.imgs[index][0]
        img_1 = self.imgs[index][1]
        img_2 = self.imgs[index][2]
        label = self.labels[index]
        img = np.array([img_0, img_1, img_2])
        img = img.transpose(1,2,0)
        img = Image.fromarray(np.uint8(img))
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

# redefine test dataset class

class Test_Dataset(Dataset):
    def __init__(self, img_test, transform=None):
        self.imgs = img_test
        self.transform = transform

    def __getitem__(self, index):
        img_0 = self.imgs[index][0]
        img_1 = self.imgs[index][1]
        img_2 = self.imgs[index][2]
        img = np.array([img_0, img_1, img_2])
        img = img.transpose(1,2,0)
        img = Image.fromarray(np.uint8(img))
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)


# load dataset
train_alldata = np.load('train.npy')
train_alldata = train_alldata.reshape(len(train_alldata), 3, 32, 32)
test_data = np.load('test.npy')
test_data = test_data.reshape(len(test_data), 3, 32, 32)
train_allcoarselabel = np.array(pd.read_csv('train1.csv'))
train_allfinelabel = np.array(pd.read_csv('train2.csv'))

# define data transformations

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    #transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    #transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    #transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# train test split 
coarseset = Train_Dataset(train_alldata, train_allcoarselabel, train_transform)
fineset = Train_Dataset(train_alldata, train_allfinelabel, train_transform)
train_coarsedata, valid_coarsedata, train_coarselabel, valid_coarselabel = train_test_split(train_alldata, train_allcoarselabel, train_size=0.9, test_size=0.1, random_state=0, shuffle=True)
train_finedata, valid_finedata, train_finelabel, valid_finelabel = train_test_split(train_alldata, train_allfinelabel, train_size=0.9, test_size=0.1, random_state=0, shuffle=True)


# create train data set
train_coarseset = Train_Dataset(train_coarsedata, train_coarselabel, train_transform)
train_fineset = Train_Dataset(train_finedata, train_finelabel, train_transform)
valid_coarseset = Valid_Dataset(valid_coarsedata, valid_coarselabel, valid_transform)
valid_fineset = Valid_Dataset(valid_finedata, valid_finelabel, valid_transform)


# train data load

train_coarseloader = Data.DataLoader(dataset=train_coarseset, batch_size=BATCH_SIZE, shuffle=True)
valid_coarseloader = Data.DataLoader(dataset=valid_coarseset, batch_size=BATCH_SIZE, shuffle=True)
train_fineloader = Data.DataLoader(dataset=train_fineset, batch_size=BATCH_SIZE, shuffle=True)
valid_fineloader = Data.DataLoader(dataset=valid_fineset, batch_size=BATCH_SIZE, shuffle=True)

#create test data loader

test_dataset = Test_Dataset(test_data, test_transform)

test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Data loading finished")

# 1.3 Net Building

# Resnet model: ResidualBlock, BottleNeck, ResNet

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # short cut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512*4, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.dropout(out)
        out = self.layer4(out)
        out = self.dropout(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = ResNet(Bottleneck, CLASS_COARSE)
model.cuda()
#params = torch.load(save_path)
#model.load_state_dict(params['model'])
loss = nn.CrossEntropyLoss()   # loss function: CrossEntropy
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) # optimizer: SGD with momentum

# set learning rate adjusting method
# scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True,
#                                                   threshold=0.01, cooldown=5, min_lr=1e-6)

'''
def lr_adjust(i):
    mult = (1 / 1e-5) ** (1/1500)
    lr_new = LR * (mult ** i)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_new
'''

# SGDR learning rate adjustment function

def SGDR(epoch):
    lr_new = optimizer.param_groups[0]["lr"]  # original learning rate
    flag = 0
    factor = 0.1 ** 0.5
    # set threshold according to learning rate
    if 0.001 < lr_new <= 0.01:
        threshold = 0.01
    elif 1e-4 < lr_new <= 0.001:
        threshold = 0.005
    elif 1e-5 < lr_new <= 1e-4:
        threshold = 0.002
    elif lr_new <= 1e-5:
        threshold = 0.001
    if epoch > 10:
        for i in range(epoch - 9, epoch + 1):
            if test_acces[i] - test_acces[epoch - 10] <= threshold:
                flag += 1
    if flag == 10:  # accuracy no growth in 10 epochs
        lr_new = lr_new * factor
        if lr_new < 1e-6:
            lr_new = LR

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new
        return True
    else:
        return False


print("Net building finished")

train_losses = []
train_acces = []
test_losses = []
test_acces = []


def train(model, train_loader, valid_loader, start_epoch):
    minloss = 100
    best_acc = 0
    best_epoch = 0
    patience = 0
    with open(program_name + ".txt", "w") as f:
        for echo in range(start_epoch, num_epochs):
            train_loss = 0  # train loss
            train_acc = 0  # train accuracy
            model.train()
            lr_adjusted = False
            ini_lr = []
            ini_loss = []
            for i, (X, label) in enumerate(train_loader):
                X = Variable(X).cuda()
                label = Variable(label).cuda()
                out = model(X)  # forwarding
                lossvalue = loss(out, label)  # loss
                optimizer.zero_grad()  # gradient to zero
                lossvalue.backward()  # backward and refresh gradient
                optimizer.step()  # optimizer run a step forward

        # calculate loss
                train_loss += float(lossvalue)
        # calculate accuracy
                _, pred = out.max(1)
                num_correct = (pred == label).sum()
                acc = int(num_correct) / X.shape[0]
                train_acc += acc
                '''
                # test initial learning rate
                print(str(i) + ": lr: " + str(optimizer.param_groups[0]["lr"])
                      + " loss: " + str(lossvalue.item()))
                ini_lr.append(optimizer.param_groups[0]["lr"])
                ini_loss.append(lossvalue.item())
                lr_adjust(i)
                #
                '''

            train_acc = round(train_acc / len(train_loader), 5)
            train_loss = round(train_loss / len(train_loader), 5)
            test_loss, test_acc = valid(model, valid_loader)
            if test_acc >= best_acc:
                if test_acc - best_acc > 0.001 or test_loss < minloss:
                    minloss = train_loss
                    best_acc = test_acc
                    best_epoch = echo
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': echo}
                    torch.save(state, save_path)

            train_losses.append(train_loss)
            train_acces.append(train_acc)
            test_losses.append(test_loss)
            test_acces.append(test_acc)
            string_echo = "echo:" + ' ' + str(echo) + ' ' + "lr:" + ' ' + str(optimizer.param_groups[0]["lr"])
            string_trainloss = "train loss:" + ' ' + str(train_loss) + ' ' + "train accuracy:" + ' ' + str(train_acc)
            string_testloss = "test loss:" + ' ' + str(test_loss) + ' ' + "test accuracy:" + ' ' + str(test_acc)
            print(string_echo)
            print(string_trainloss + '  ' + string_testloss)
            f.write(string_echo + ' ' + string_trainloss + ' ' + string_testloss + '\n')
            # adjust learning rate according to epochs
            patience += 1
            if patience >= 10:
                lr_adjusted = SGDR(echo)
            if lr_adjusted:
                patience = 0
        print("training finished.")
        print("best epoch:" + ' ' + str(best_epoch))
        print("minimum training loss:" + ' ' + str(minloss))
        print("best training accuracy:" + ' ' + str(best_acc))


def valid(model, valid_loader):
    model.eval()
    test_acc = 0.0
    test_loss = 0.0
    for i, (X, label) in enumerate(valid_loader):
        X = Variable(X).cuda()
        label = Variable(label).cuda()
        out = model(X)
        lossvalue = loss(out, label)
        # calculate test loss
        test_loss += float(lossvalue)
        # calculate test accuracy
        _, pred = out.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / X.shape[0]
        test_acc += acc

    test_acc = round(test_acc / len(valid_loader), 5)
    test_loss = round(test_loss / len(valid_loader), 5)
    return test_loss, test_acc


def test(model, test_loader):
    model.eval()
    pred_all = None
    for i, img in enumerate(test_loader):
        img = img.type(torch.FloatTensor)
        img = Variable(img).cuda()
        output = model(img)
        _, pred = output.max(1)
        if pred_all is None:
            pred_all = torch.cat([pred])
        else:
            pred_all = torch.cat([pred_all, pred])
    y_pred = pred_all.cpu().detach().numpy()
    index = np.linspace(0, 9999, 10000, dtype=int)
    result = pd.DataFrame({'image_id': index, 'label': y_pred})
    result.to_csv('samplesummission1.csv', index=False, sep=',')


if __name__ == "__main__":
    # train
    train(model, train_coarseloader, valid_coarseloader, start_epoch)
    # load trained model
    params = torch.load(save_path)
    model.load_state_dict(params['model'])
    # test
    test(model, test_loader)




