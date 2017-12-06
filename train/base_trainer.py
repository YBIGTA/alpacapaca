import time
import math
import pickle 
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable

class LetterTrainer():

    def __init__(self, model, data_reader, criterion=nn.NLLLoss(), learning_rate=0.0005,
                n_iters=100000, print_every=5000, plot_every=500):

        self.rnn = model
        self.data_reader = data_reader
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.print_every = print_every
        self.plot_every = plot_every
        self.i = 0

    def train(self, input_line_tensor, target_line_tensor):

        self.hidden = self.rnn.initHidden()

        self.rnn.zero_grad()

        self.loss = 0

        for i in range(input_line_tensor.size()[0]):
            self.output, self.hidden = self.rnn(input_line_tensor[i], self.hidden)
            self.loss += self.criterion(self.output, target_line_tensor[i])

        self.loss.backward()

        for p in self.rnn.parameters():
            p.data.add_(-self.learning_rate, p.grad.data)

        return self.output, self.loss.data[0] / input_line_tensor.size()[0]

    def run(self):

        self.all_losses = []
        self.total_loss = 0 # Reset every plot_every iters

        start = time.time()

        for i in range(self.i, self.n_iters + 1):
            output, loss = self.train(*self.data_reader.randomTrainingExample())
            self.total_loss += loss

            if i % self.print_every == 0:
                print('%s (%d %d%%) %.4f' % (self.timeSince(start), i, i / self.n_iters * 100, loss))
                
            if i % self.plot_every == 0:
                self.all_losses.append(self.total_loss / self.plot_every)
                self.total_loss = 0

            self.i = i

    @staticmethod
    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

class DataLoaderTrainer():

    def __init__(self, model, dataloader, optimizer, scheduler, criterion=nn.MSELoss(), 
                print_every=1, plot_every=1):

        self.rnn = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.print_every = print_every
        self.plot_every = plot_every
        self.i = 0
        self.all_losses = []

    def train(self):

        self.losses = []
        for inputs, targets in tqdm(self.dataloader):

            self.inputs, self.targets = inputs, targets

            self.optimizer.zero_grad()
            self.rnn.hidden = self.rnn.init_hidden()

            self.sentence_out = self.rnn(inputs)
            loss = self.criterion(self.sentence_out, self.targets)
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.data)

        return sum(self.losses) / len(self.losses)

    def run(self, epochs=10):

        start = time.time()

        for i in range(self.i, epochs):
            self.i = i

            self.avg_loss = self.train()[0]

            if i % self.print_every == 0:
                print('%s (%d %d%%) %.4f' % (self.timeSince(start), i, i / epochs * 100, self.avg_loss))
                
            if i % self.plot_every == 0:
                self.all_losses.append(self.avg_loss)

    @staticmethod
    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)    