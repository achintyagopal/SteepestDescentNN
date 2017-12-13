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

        self.w1 = nn.Parameter(torch.randn(784, 400), requires_grad=True)
        self.w2 = nn.Parameter(torch.randn(400, 10), requires_grad=True)
        self.lr = nn.Parameter(torch.randn(1), requires_grad=True)

        for p in self.parameters():
            p.data.uniform_(-0.25,0.25)

        self.lr.data[0] = 1e-1
        self.relu = nn.ReLU()
        self.isConvert = False

    def encode(self, x):
        if not self.isConvert:
            h1 = self.relu(x.mm(self.w1))
            return h1.mm(self.w2)
        else:
            h1 = self.relu(x.mm(self.w1 - self.lr * self.g1))
            return h1.mm(self.w2 - self.lr * self.g2)

    def forward(self, x):
        return self.encode(x.view(-1, 784))

    def learnLR(self, count):

        self.g1 = self.w1.grad.detach() / count
        self.g1.volatile = False
        self.g2 = self.w2.grad.detach() / count
        self.g2.volatile = False
        self.count = count
        # self.w3 = self.w1.detach()
        # self.w3.requires_grad = False
        # self.w4 = self.w2.detach()
        # self.w4.requires_grad = False

        self.lr.data[0] = 1e-1
        self.isConvert = True

    def updateParams(self, firstDeriv, secondDeriv, count):
        self.lr.data.sub_(firstDeriv / secondDeriv)
        # self.lr.data.sub_(firstDeriv * 1e-1 / count)
        # self.lr.data.sub_(firstDeriv / secondDeriv)
        # print('lr', self.lr.data[0])
        # self.w1.data.sub_(1e-1 * self.w1.grad.data)
        # self.w2.data.sub_(1e-1 * self.w2.grad.data)

        self.w1.data.sub_((self.lr * self.g1).data)
        self.w2.data.sub_((self.lr * self.g2).data)
        self.isConvert = False

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
        model.updateParams(firstDeriv, secondDeriv, 1)
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

        if batch_idx % args.log_interval == 0:
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

# model = VAE()
# lossFn = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=1e-1)

# def preprocess():
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, y_class) in enumerate(train_loader):

#         optimizer.zero_grad()

#         data = Variable(data)
#         y_class = Variable(y_class)

#         y_pred = model(data)
#         loss = lossFn(y_pred, y_class)
#         train_loss += loss
#         loss.backward()

#         optimizer.step()

#         if batch_idx >= 10:
#             break

#         if batch_idx % args.log_interval == 0:
#             print('Preproces: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.data[0] / len(data)))


# def train(epoch):
#     model.train()
#     train_loss = 0

#     secondDeriv = 0.
#     firstDeriv = 0.
#     count = 0
#     for batch_idx, (data, y_class) in enumerate(train_loader):

#         # if (batch_idx + 1) % 2 == 0:
#         # if model.isConvert and count == 1:
#         #     # model.lr.data.sub_(firstDeriv / secondDeriv)
#         #     model.updateParams(firstDeriv, secondDeriv, count)
#         #     count = 0
#         #     optimizer.zero_grad()
#         # elif not model.isConvert and count == 20:
#         #     model.learnLR(count)
#         #     firstDeriv = 0
#         #     secondDeriv = 0
#         #     count = 0
#         #     optimizer.zero_grad()

#         optimizer.zero_grad()

#         data = Variable(data)
#         y_class = Variable(y_class)
#         y_pred = model(data)
#         loss = lossFn(y_pred, y_class)
#         train_loss += loss
#         loss.backward()
#         count += 1

#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.data[0] / len(data)))

#     model.learnLR(count)
#     firstDeriv = 0
#     secondDeriv = 0
#     count = 0
#     for batch_idx, (data, y_class) in enumerate(train_loader):

#         data = Variable(data)
#         y_class = Variable(y_class)

#         loss = lossFn(model(data), y_class)
#         grad = torch.autograd.grad(loss, [model.lr], create_graph=True)[0]
#         firstDeriv += grad.data

#         if model.lr.grad is not None:
#             model.lr.grad.data.zero_()
#         grad.backward()
#         secondDeriv += model.lr.grad.data
#         count += 1

#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.data[0] / len(data)))

#     model.updateParams(firstDeriv, secondDeriv, count)
#         # count += 1
#         # totalLoss = loss / 100

#         # if model.isConvert:
#         #     # loss.backward()
#         #     grad_params = torch.autograd.grad(loss, [model.lr], create_graph=True)

#         #     firstDeriv += grad_params[0].data

#         #     if model.lr.grad is not None:
#         #         model.lr.grad.data.zero_()
#         #     grad_params[0].backward()
#         #     secondDeriv += model.lr.grad.data

#         # else:
#         #     loss.backward()

#         # if batch_idx % 1 == 0:
#         #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#         #         epoch, batch_idx * len(data), len(train_loader.dataset),
#         #         100. * batch_idx / len(train_loader),
#         #         loss.data[0] / len(data)))

#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#           epoch, train_loss.data[0] / len(train_loader)))

# def test(epoch):
#     model.eval()
#     # test_loss = 0
#     correct = 0
#     total = 0
#     print(model.lr)
#     for batch_idx, (data, y_class) in enumerate(test_loader):

#         data = Variable(data, volatile=True)
#         y_pred = model(data)
#         # test_loss += loss_function(y_class, y_pred)
#         z = y_pred.data.cpu().numpy()
#         for i, row in enumerate(z):
#             pred = np.argmax(row)
#             if pred == y_class[i]:
#                 correct += 1
#         total += len(y_class)

#     # test_loss /= len(test_loader.dataset)
#     print('Correct: ' + str(correct))
#     print('Total: ' + str(total))
#     # print('====> Test set loss: ' + str(test_loss.data.cpu().numpy()[0]))

# def stack(ra):
#     num_per_row = int(np.sqrt(len(ra)))
#     rows = [np.concatenate(tuple(ra[i* num_per_row : i*num_per_row + num_per_row]), axis=1) 
#             for i in range(num_per_row)]
#     img = np.concatenate(tuple(rows), axis=0)
#     return img

# if args.load_model:
#     model = torch.load(args.load_model)

# preprocess()
# for epoch in range(1, args.epochs + 1):
    
#     if "train" in args.mode:
#         train(epoch)
#     if "eval" in args.mode:
#         test(epoch)

#     if epoch % args.save_interval == 0:
#         torch.save(model, args.save_model + "_" + str(epoch))


