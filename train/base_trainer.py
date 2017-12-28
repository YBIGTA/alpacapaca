import time
import math
import pickle 
from tqdm import tqdm
import logging, os
from datetime import datetime

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
                print_every=1, plot_every=1, log_dirpath='train/log/', log_filename='{datetime}.log', use_gpu=False):

        self.rnn = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.print_every = print_every
        self.plot_every = plot_every
        self.i = 0
        self.all_losses = []
        self.use_gpu = use_gpu
        
        # 프로젝트 최상단 디렉토리 경로를 가져온다.
        dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.abs_output_dirpath = os.path.join(dirname, log_dirpath)
        if not os.path.exists(self.abs_output_dirpath):
            os.mkdir(self.abs_output_dirpath)
            
        log_filepath = os.path.join(self.abs_output_dirpath, log_filename)

        self.logger = logging.getLogger('pacapaca_logger')
        
        if not self.logger.handlers: # execute only if logger doesn't exist:
            fileHandler = logging.FileHandler(log_filepath.format(datetime=datetime.now()))
            streamHandler = logging.StreamHandler(os.sys.stdout)

            formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s', datefmt='%m-%d %H:%M:%S')

            fileHandler.setFormatter(formatter)
            streamHandler.setFormatter(formatter)

            self.logger.addHandler(fileHandler)
            self.logger.addHandler(streamHandler)
            self.logger.setLevel(logging.INFO)
        
        self.base_message = ("Epoch: {epoch:<3d} Progress: {progress:<.1%} "
                             "Loss: {loss:<.6} "
                             "Learning rate: {learning_rate:<.4} "
                             "Elapsed: {elapsed} ")

    def train(self):

        self.losses = []
        for inputs, targets in tqdm(self.dataloader):

            if self.use_gpu:
                self.inputs, self.targets = Variable(inputs.cuda()), Variable(targets.cuda())
            else:
                self.inputs, self.targets = Variable(inputs), Variable(targets)

            self.optimizer.zero_grad()
            self.rnn.init_hidden(inputs.size(0), self.use_gpu)
            self.sentence_out = self.rnn(self.inputs)
            loss = self.criterion(self.sentence_out, self.targets)
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.data[0])

            if self.i == 0: # for testing
                break

        return sum(self.losses) / len(self.losses)

    def run(self, epochs=10):

        start = time.time()

        for i in range(self.i, epochs + 1):
            self.i = i

            self.avg_loss = self.train()
            self.scheduler.step(self.avg_loss)

            if i % self.print_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                message = self.base_message.format(epoch=i, progress=i/epochs, 
                                                   loss=self.avg_loss, 
                                                   learning_rate=current_lr,
                                                   elapsed=self.timeSince(start))
                self.logger.info(message)
                
            if i % self.plot_every == 0:
                self.all_losses.append(self.avg_loss)

    @staticmethod
    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    
class TrainValTrainer():

    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler, criterion=nn.MSELoss(), 
                print_every=1, plot_every=1, log_dirpath='train/log/', log_filename='{datetime}.log', use_gpu=False):

        self.rnn = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.print_every = print_every
        self.plot_every = plot_every
        self.i = 0
        self.all_losses = []
        self.use_gpu = use_gpu
        
        # 프로젝트 최상단 디렉토리 경로를 가져온다.
        dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.abs_output_dirpath = os.path.join(dirname, log_dirpath)
        if not os.path.exists(self.abs_output_dirpath):
            os.mkdir(self.abs_output_dirpath)
            
        log_filepath = os.path.join(self.abs_output_dirpath, log_filename)

        self.logger = logging.getLogger('pacapaca_logger')
        
        if not self.logger.handlers: # execute only if logger doesn't exist:
            fileHandler = logging.FileHandler(log_filepath.format(datetime=datetime.now()))
            streamHandler = logging.StreamHandler(os.sys.stdout)

            formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s', datefmt='%m-%d %H:%M:%S')

            fileHandler.setFormatter(formatter)
            streamHandler.setFormatter(formatter)

            self.logger.addHandler(fileHandler)
            self.logger.addHandler(streamHandler)
            self.logger.setLevel(logging.INFO)
        
        self.base_message = ("Epoch: {epoch:<3d} Progress: {progress:<.1%} ({elapsed}) "
                             "Loss: {loss:<.6} "
                             "Val Loss: {val_loss:<.6} "
                             "Learning rate: {learning_rate:<.4} "
                            )

    def train(self):

        self.losses = []
        for inputs, targets in tqdm(self.train_dataloader):

            if self.use_gpu:
                self.inputs, self.targets = Variable(inputs.cuda()), Variable(targets.cuda())
            else:
                self.inputs, self.targets = Variable(inputs), Variable(targets)

            self.optimizer.zero_grad()
            self.rnn.init_hidden(inputs.size(0), self.use_gpu)
            self.sentence_out = self.rnn(self.inputs)
            loss = self.criterion(self.sentence_out, self.targets)
                        
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.data[0])
            
            if self.i == 0: # for testing
                break

        self.val_losses = []
        for val_inputs, val_targets in self.val_dataloader:
            if self.use_gpu:
                self.val_inputs, self.val_targets = Variable(val_inputs.cuda()), Variable(val_targets.cuda())
            else:
                self.val_inputs, self.val_targets = Variable(val_inputs), Variable(val_targets)

            self.rnn.init_hidden(val_inputs.size(0), self.use_gpu)
            self.val_sentence_out = self.rnn(self.val_inputs)
            val_loss = self.criterion(self.val_sentence_out, self.val_targets)
            self.val_losses.append(val_loss.data[0])
        
        avg_loss = sum(self.losses) / len(self.losses)
        val_avg_loss = sum(self.val_losses) / len(self.val_losses)
        return avg_loss, val_avg_loss

    def run(self, epochs=10):

        start = time.time()

        for i in range(self.i, epochs + 1):
            self.i = i

            self.avg_loss, self.avg_val_loss = self.train()
            self.scheduler.step(self.avg_loss)

            if i % self.print_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                message = self.base_message.format(epoch=i, progress=i/epochs, 
                                                   loss=self.avg_loss, 
                                                   val_loss=self.avg_val_loss,
                                                   learning_rate=current_lr,
                                                   elapsed=self.timeSince(start))
                self.logger.info(message)
                
            if i % self.plot_every == 0:
                self.all_losses.append(self.avg_loss)

    @staticmethod
    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)    