from __future__ import print_function
import numpy as np
import math
import sys
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
import cv2 as cv

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-level', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--sample-size', type=int, default=100, metavar='N',
                    help='number of samples to generate (should be perfect square)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument("--load-model", type=str,
        help="The file containing already trained model.")
parser.add_argument("--save-model", default="dense_mnist", type=str,
        help="The file containing already trained model.")
parser.add_argument("--mode", type=str, default="train-eval", choices=["train", "eval", "train-eval"],
                        help="Operating mode: train and/or test.")
args = parser.parse_args()


torch.manual_seed(args.seed)

kwargs = {}

if "train" in args.mode:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

if "eval" in args.mode:
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

if args.mode == "eval":
    if not args.load_model:
        raise ValueError("Need which model to evaluate")
    args.epoch = 1
    args.eval_interval = 1


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'mine1': [32,'M2',32,'M2',32,'M2'],
    'mine2': [32,'M8'],
}


# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Linear(512, 10)
#         self.softmax = nn.Softmax()

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.softmax(self.classifier(out))
#         return out

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M2':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             elif x == 'M4':
#                 layers += [nn.MaxPool2d(kernel_size=4, stride=4)]
#             elif x == 'M8':
#                 layers += [nn.MaxPool2d(kernel_size=8, stride=8)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

# model = VGG('mine2')

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # self.fc1 = nn.Sequential(
        #     # nc * 32 * 32
        #     nn.Conv2d(3, 32, 4, 4, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     # 32 x 8 x 8
        #     nn.Conv2d(32, 32, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.2, inplace=True)

        #     # 32 x 4 x 4
        # )

        # self.fc2 = nn.Linear(512, 10)

        # self.fc1 = nn.Sequential(
        #     # state size. (ndf) x 32 x 32
        #     nn.Conv2d(3, 16 * 2, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (16*2) x 16 x 16
        #     nn.Conv2d(16 * 2, 16 * 4, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (16*4) x 8 x 8
        #     nn.Conv2d(16 * 4, 16 * 8, 4, 2, 1, bias=False)
        # )
        # self.fc1 = nn.Linear(32*32*3, 2048)

        # self.fc2 = nn.Linear(2048, 10)
        # self.fc3 = nn.Linear(512, 10)

        self.relu = nn.ReLU()
        self.weight1 = nn.Parameter(torch.Tensor(
                16*2, 3 // 1, 4,4))
        self.weight2 = nn.Parameter(torch.Tensor(
                16 * 4, 16 * 2, 4,4))
        self.weight3 = nn.Parameter(torch.Tensor(
                16 * 8, 16 * 4, 4,4))

        self.lr = nn.Parameter(torch.randn(1), requires_grad=True)
        self.lr.data[0] = 1e-2

        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(16 * 2)
        self.bn2 = nn.BatchNorm2d(16 * 4)
        self.bn3 = nn.BatchNorm2d(16 * 5)

        self.bias = nn.Parameter(torch.Tensor(1, 10))
        self.fc1 = nn.Parameter(torch.Tensor(2048, 10))

        stdv = 1. / math.sqrt(3*4*4)
        for p in self.parameters():
            p.data.uniform_(-stdv, stdv)

        self.isConvert = False

    def encode(self, x):
        if not self.isConvert:
            h1 = self.leakyrelu(self.bn1(F.conv2d(x, self.weight1, stride=2, padding=1)))
            h2 = self.leakyrelu(self.bn2(F.conv2d(h1, self.weight2, stride=2, padding=1)))
            h3 = F.conv2d(h2, self.weight3, stride=2, padding=1)
            h3 = h3.view(-1, 2048)
            return h3.mm(self.fc1) + self.bias
        else:
            h1 = self.leakyrelu(self.bn1(F.conv2d(x, self.weight4 - self.lr * self.g1, stride=2, padding=1)))
            h2 = self.leakyrelu(self.bn2(F.conv2d(h1, self.weight5 - self.lr * self.g2, stride=2, padding=1)))
            h3 = F.conv2d(h2, self.weight6 - self.lr * self.g3, stride=2, padding=1)
            h3 = h3.view(-1, 2048)
            return h3.mm(self.fc2 - self.lr * self.g4) + self.bias2 - self.lr * self.g5

    def forward(self, x):
        return self.encode(x)

    def learnLR(self, count):

        self.g1 = self.weight1.grad.detach() / count
        self.g1.volatile = False

        self.g2 = self.weight2.grad.detach() / count
        self.g2.volatile = False

        self.g3 = self.weight3.grad.detach() / count
        self.g3.volatile = False

        self.g4 = self.fc1.grad.detach() / count
        self.g4.volatile = False

        self.g5 = self.bias.grad.detach() / count
        self.g5.volatile = False

        self.weight4 = self.weight1.detach()
        self.weight4.requires_grad = False

        self.weight5 = self.weight2.detach()
        self.weight5.requires_grad = False

        self.weight6 = self.weight3.detach()
        self.weight6.requires_grad = False

        self.fc2 = self.fc1.detach()
        self.fc2.requires_grad = False

        self.bias2 = self.bias.detach()
        self.bias2.requires_grad = False

        self.lr.data[0] = 1e-2
        self.isConvert = True
        # print(self.g1)
        # return self.w1.grad.clone()


    def updateParams(self, firstDeriv, secondDeriv):
        self.isConvert = False

        if firstDeriv[0] / secondDeriv[0] > self.lr.data[0]:
            return
        self.lr.data.sub_(firstDeriv / secondDeriv)
        print(self.lr)
        self.weight1.data.sub_((self.lr * self.g1).data)
        self.weight2.data.sub_((self.lr * self.g2).data)
        self.weight3.data.sub_((self.lr * self.g3).data)
        self.fc1.data.sub_((self.lr * self.g4).data)
        self.bias.data.sub_((self.lr * self.g5).data)

model = VAE()
lossFn = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=1e-1)

def preprocess():
    model.train()
    train_loss = 0
    for batch_idx, (data, y_class) in enumerate(train_loader):

        optimizer.zero_grad()

        data = Variable(data)
        y_class = Variable(y_class)

        y_pred = model(data)
        loss = lossFn(y_pred, y_class)
        train_loss += loss
        loss.backward()

        optimizer.step()

        if batch_idx >= 10:
            break

        if batch_idx % args.log_interval == 0:
            print('Preproces: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))


def train(epoch):
    model.train()
    train_loss = 0

    secondDeriv = 0.
    firstDeriv = 0.
    count = 0
    for batch_idx, (data, y_class) in enumerate(train_loader):

        # if (batch_idx + 1) % 2 == 0:
        # if model.isConvert and count == 1:
        #     # model.lr.data.sub_(firstDeriv / secondDeriv)
        #     model.updateParams(firstDeriv, secondDeriv, count)
        #     count = 0
        #     optimizer.zero_grad()
        # elif not model.isConvert and count == 20:
        #     model.learnLR(count)
        #     firstDeriv = 0
        #     secondDeriv = 0
        #     count = 0
        #     optimizer.zero_grad()

        optimizer.zero_grad()

        data = Variable(data)
        y_class = Variable(y_class)
        y_pred = model(data)
        loss = lossFn(y_pred, y_class)
        train_loss += loss
        loss.backward()

        model.learnLR(1)
        loss = lossFn(model(data), y_class)
        grad = torch.autograd.grad(loss, [model.lr], create_graph=True)[0]
        firstDeriv = grad.data

        if model.lr.grad is not None:
            model.lr.grad.data.zero_()
        grad.backward()
        secondDeriv = model.lr.grad.data
        model.updateParams(firstDeriv, secondDeriv)
        # count += 1
        # totalLoss = loss / 100

        # if model.isConvert:
        #     # loss.backward()
        #     grad_params = torch.autograd.grad(loss, [model.lr], create_graph=True)

        #     firstDeriv += grad_params[0].data

        #     if model.lr.grad is not None:
        #         model.lr.grad.data.zero_()
        #     grad_params[0].backward()
        #     secondDeriv += model.lr.grad.data

        # else:
        #     loss.backward()

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss.data[0] / len(train_loader)))

def test(epoch):
    model.eval()
    # test_loss = 0
    correct = 0
    total = 0
    print(model.lr)
    for batch_idx, (data, y_class) in enumerate(test_loader):

        data = Variable(data, volatile=True)
        y_pred = model(data)
        # test_loss += loss_function(y_class, y_pred)
        z = y_pred.data.cpu().numpy()
        for i, row in enumerate(z):
            pred = np.argmax(row)
            if pred == y_class[i]:
                correct += 1
        total += len(y_class)

    # test_loss /= len(test_loader.dataset)
    print('Correct: ' + str(correct))
    print('Total: ' + str(total))
    # print('====> Test set loss: ' + str(test_loss.data.cpu().numpy()[0]))

def stack(ra):
    num_per_row = int(np.sqrt(len(ra)))
    rows = [np.concatenate(tuple(ra[i* num_per_row : i*num_per_row + num_per_row]), axis=1) 
            for i in range(num_per_row)]
    img = np.concatenate(tuple(rows), axis=0)
    return img

if args.load_model:
    model = torch.load(args.load_model)

# preprocess()
for epoch in range(1, args.epochs + 1):
    
    if "train" in args.mode:
        train(epoch)
    if "eval" in args.mode:
        test(epoch)

    if epoch % args.save_interval == 0:
        torch.save(model, args.save_model + "_" + str(epoch))

