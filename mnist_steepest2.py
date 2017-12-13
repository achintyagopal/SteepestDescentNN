from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import cv2 as cv

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--sample-size', type=int, default=100, metavar='N',
                    help='number of samples to generate (should be perfect square)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=1000, metavar='N',
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
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

if "eval" in args.mode:
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

if args.mode == "eval":
    if not args.load_model:
        raise ValueError("Need which model to evaluate")
    args.epoch = 1
    args.eval_interval = 1


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # self.w1 = nn.Parameter(torch.randn(784, 400), requires_grad=True)
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 10)
        # self.w2 = nn.Parameter(torch.randn(400, 10), requires_grad=True)
        self.lr = nn.Parameter(torch.randn(1), requires_grad=True)
        self.lr.data[0] = 1e-1

        self.relu = nn.ReLU()
        self.isConvert = False

    def encode(self, x):
        if not self.isConvert:
            return self.fc2(self.relu(self.fc1(x)))
        else:

            # for i, p in enumerate(self.parameters()):
                # if i in self.grads:
                    # p -= self.lr * self.grads[i]
                    # p.data.sub_((self.lr * self.grads[i]).data)
            self.fc3 = nn.Linear(784, 400)
            self.fc3.weight.requires_grad = False
            self.fc3.weight -= self.lr * self.g1
            h1 = self.fc2(self.relu(self.fc1(x)))
            # for i, p in enumerate(self.parameters()):
                # if i in self.grads:
                    # p.data.add_((self.lr * self.grads[i]).data)
            print(self.lr.grad)
            return h1

    def forward(self, x):
        if not self.isConvert:
            return self.fc2(self.relu(self.fc1(x.view(-1, 784))))
        else:

            # for i, p in enumerate(self.parameters()):
                # if i in self.grads:
                    # p -= self.lr * self.grads[i]
                    # p.data.sub_((self.lr * self.grads[i]).data)
            self.fc3 = nn.Linear(784, 400)
            # self.fc3.weight.requires_grad = False
            self.fc3.weight = self.fc3.weight.detach() - self.lr * self.g1

            self.fc4 = nn.Linear(400, 10)
            self.fc4.weight.requires_grad = False
            self.fc4.weight -= self.lr * self.g2



            h1 = self.fc2(self.relu(self.fc3(x.view(-1, 784))))
            # for i, p in enumerate(self.parameters()):
                # if i in self.grads:
                    # p.data.add_((self.lr * self.grads[i]).data)
            print(self.lr.grad)
            return h1

    def convert(self):
        # self.grads = {}
        # for i, p in enumerate(self.parameters()):
            # print(p)
            # if p.grad is not None:
                # grad = p.grad.detach()
                # grad.volatile = False
                # self.grads[i] = grad
        self.g1 = self.fc1.weight.detach()
        self.g1.volatile = False
        # self.g1 = self.g1.data

        self.g2 = self.fc2.weight.detach()
        self.g2.volatile = False
        # self.g2 = self.g2.data

        self.g3 = self.fc1.bias.detach()
        self.g3.volatile = False
        # self.g3 = self.g3.data

        self.g4 = self.fc2.bias.detach()
        self.g4.volatile = False
        # self.g4 = self.g4.data

        self.isConvert = True
        # self.g1 = self.w1.grad.detach()
        # self.g1.volatile = False
        # self.g2 = self.w2.grad.detach()
        # self.g2.volatile = False

        # self.fc3 = self.fc1.detach()
        # self.w3.requires_grad = False
        # self.w4 = self.w2.detach()
        # self.w4.requires_grad = False
        # self.isConvert = True
        # print(self.g1)
        # return self.w1.grad.clone()



model = VAE()
loss = nn.CrossEntropyLoss()

def loss_function(y_class, y_pred):
    return loss(y_pred, Variable(y_class))

optimizer = optim.SGD(model.parameters(), lr=1e-1)
optimizer2 = optim.SGD([model.lr], lr=1e-1)

def train(epoch):
    model.train()
    train_loss = 0

    # 1/3 regular SGD
    # 1/3 total gradient
    # 1/3, copy gradient and copy model without grad
    #      new parameter with gradient, forward prop = weights - parameter * gradient
    #      SGD for parameter
    for batch_idx, (data, y_class) in enumerate(train_loader):
        # optimizer.zero_grad()
        # data = Variable(data)

        # y_pred = model(data)
        # loss = loss_function(y_class, y_pred)
        # train_loss += loss
        # loss.backward()

        model.convert()
        # optimizer.step()

        # train_loss += loss.data[0]
        if batch_idx < len(train_loader) / 3:
            optimizer.zero_grad()
            data = Variable(data)

            y_pred = model(data)
            loss = loss_function(y_class, y_pred)
            train_loss += loss
            loss.backward()

            optimizer.step()
        elif batch_idx == int(len(train_loader) / 3):

            optimizer.zero_grad()
            train_loss = 0
            data = Variable(data)

            y_pred = model(data)
            loss = loss_function(y_class, y_pred)
            train_loss += loss
            # loss.backward()
        elif batch_idx < 2 * len(train_loader) / 3:
            data = Variable(data)

            y_pred = model(data)
            loss = loss_function(y_class, y_pred)
            train_loss += loss
            # loss.backward()
        elif batch_idx == int(2 * len(train_loader) / 3):
            optimizer.zero_grad()
            train_loss /= len(train_loader) / 3
            train_loss.backward()

            model.convert()
            optimizer2.zero_grad()
            train_loss = 0
            data = Variable(data)

            y_pred = model(data)
            loss = loss_function(y_class, y_pred)
            train_loss += loss
            # loss.backward()

            # optimizer2.step()
        else:
            # optimizer2.zero_grad()
            data = Variable(data)

            y_pred = model(data)
            loss = loss_function(y_class, y_pred)
            train_loss += loss
            # loss.backward()

            # optimizer2.step()


        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))
    optimizer2.zero_grad()
    
    train_loss /= len(train_loader) / 3
    train_loss.backward()
    optimizer2.step()

    train_loss /= len(train_loader.dataset)
    # train_loss.backward()
    # optimizer.zero()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    print(model.lr)
    for batch_idx, (data, y_class) in enumerate(test_loader):

        data = Variable(data, volatile=True)
        y_pred = model(data)
        test_loss += loss_function(y_class, y_pred)
        z = y_pred.data.cpu().numpy()
        for i, row in enumerate(z):
            pred = np.argmax(row)
            if pred == y_class[i]:
                correct += 1
        total += len(y_class)

    test_loss /= len(test_loader.dataset)
    print('Correct: ' + str(correct))
    print('Total: ' + str(total))
    print('====> Test set loss: ' + str(test_loss.data.cpu().numpy()[0]))

def stack(ra):
    num_per_row = int(np.sqrt(len(ra)))
    rows = [np.concatenate(tuple(ra[i* num_per_row : i*num_per_row + num_per_row]), axis=1) 
            for i in range(num_per_row)]
    img = np.concatenate(tuple(rows), axis=0)
    return img

if args.load_model:
    model = torch.load(args.load_model)

for epoch in range(1, args.epochs + 1):
    
    if "train" in args.mode:
        train(epoch)
    if "eval" in args.mode:
        test(epoch)

    if epoch % args.save_interval == 0:
        torch.save(model, args.save_model + "_" + str(epoch))

