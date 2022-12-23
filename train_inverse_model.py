import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import csv
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import os
import pdb
import time
from itertools import permutations
from absl import app
from absl import flags
from dataloaders.gibson import GibsonDatasetPair

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('bottleneck_size', 3, 'Output dimension of CNN')
flags.DEFINE_float('lr', 0.0001, 'Learning rate')
flags.DEFINE_float('lr_decay', 0.1, 'Learning rate decay gamma')
flags.DEFINE_float('lr_decay_every', 200, 'Learning rate decay rate')
flags.DEFINE_float('weight_decay', 0.0, 'Weight decay in optimizer')
flags.DEFINE_integer('gpu', 0, 'Which GPU to use.')
flags.DEFINE_string('logdir', 'debug', 'Name of tensorboard logdir')

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # resnet
        self.resnet18 = models.resnet18(pretrained=True)
        self.modules = list((self.resnet18).children())[:-2]    # converts [batch_size, 1000] to [batch_size, 512, 7, 7]
        self.resnet18 = nn.Sequential(*self.modules)

        # freeze resnet model
        (self.resnet18).eval()
        for param in (self.resnet18).parameters():
            param.requires_grad = False

        # CNN
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64*3*3, 128)
        self.fc2 = nn.Linear(128, FLAGS.bottleneck_size)

        # accuracy_learned_fc
        self.fc_accuracy = nn.Linear(FLAGS.bottleneck_size, 3)


    def forward(self, k, k_plus_one):

        # freeze resnet
        self.resnet18.eval()

        # resnet
        resnet_k = self.resnet18(k)
        resnet_k_plus_one = self.resnet18(k_plus_one)
        resnet_output = torch.cat([resnet_k, resnet_k_plus_one], dim=1)

        # CNN
        x = self.conv1(resnet_output)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)

        # accuracy_learned_fc
        y = self.fc_accuracy(x)                                               # [batch_size, 3]                          
        return y


def train(model, device, train_loader, optimizer, epoch, val_loader, writer, iteration):

    model.train()

    print_every = 100
    iteration_ = iteration

    train_loss = 0
    train_acc = 0

    for batch_idx, (be, ae, act, rew, term, gt) in enumerate(train_loader):

        be, ae, act = be.to(device), ae.to(device), act.to(device)
        optimizer.zero_grad()

        # forward pass 
        y = model(be, ae)

        # losses
        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(y, act)
        train_loss += loss.item()

        # accuracy_learned_fc
        pred = y.argmax(dim=1, keepdim=True)        # get the index of the max log-probability
        train_acc += pred.eq(act.view_as(pred)).sum().item()

        # backprop
        loss.backward()
        optimizer.step()


        if batch_idx % print_every == 0 and batch_idx != 0:

            iteration_ += 1

            train_loss /= print_every
            train_acc /= ((print_every * FLAGS.batch_size) + FLAGS.batch_size)
            val_loss, val_acc = validate(model, device, val_loader, train_loader, 25)
            model.train()

            print("iter: ", iteration_, ", train_loss: ", train_loss, ", val_loss: ", val_loss.item())

            # save to tensorboard
            writer.add_scalar('Loss/train', train_loss, iteration_)
            writer.add_scalar('Loss/val', val_loss, iteration_)
            writer.add_scalar('Accuracy/train', train_acc, iteration_)
            writer.add_scalar('Accuracy/val', val_acc, iteration_)

            # save model
            file_name = os.path.join('inverse_model_runs/', FLAGS.logdir, 'model-{:d}.pth'.format(iteration_))
            torch.save(model.state_dict(), file_name)

            # reset
            train_loss = 0
            train_acc = 0

    return iteration_



def validate(model, device, val_loader, train_loader, print_every):

    model.eval()
    val_loss = 0
    val_acc = 0
    j = 0

    with torch.no_grad():
        for batch_idx, (be, ae, act, rew, term, gt) in enumerate(val_loader):

            be, ae, act = be.to(device), ae.to(device), act.to(device)

            # forward pass
            y = model(be, ae)

            # losses
            cross_entropy_loss = nn.CrossEntropyLoss()
            val_loss += cross_entropy_loss(y, act)

            # accuracy
            pred = y.argmax(dim=1, keepdim=True)        # get the index of the max log-probability
            val_acc += pred.eq(act.view_as(pred)).sum().item()

            j += 1
            if j == print_every:
                break

    val_loss /= print_every
    val_acc /= ((print_every * FLAGS.batch_size) + FLAGS.batch_size)

    return val_loss, val_acc


def main(argv):

    torch.cuda.set_device(FLAGS.gpu)

    train_dataset = GibsonDatasetPair('data/inverse_model/medium_inverse_train_40k_data.npy')
    train_loader = data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=2)
    val_dataset = GibsonDatasetPair('data/inverse_model/medium_inverse_val_data.npy')
    val_loader = data.DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=True)

    device = torch.device("cuda")
    model_ = model().to(device)
    optimizer = torch.optim.Adam(model_.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    writer = SummaryWriter(str('inverse_model_runs/' + str(FLAGS.logdir)))
    scheduler = StepLR(optimizer, step_size=FLAGS.lr_decay_every, gamma=FLAGS.lr_decay)
    iteration = 0

    for epoch in range(1, 200):
        print("Train Epoch: ", epoch)
        iteration = train(model_, device, train_loader, optimizer, epoch, val_loader, writer, iteration)
        scheduler.step()


if __name__ == '__main__':
    app.run(main)
