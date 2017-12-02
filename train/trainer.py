import re
import random
from collections import Counter
import time
import math
import pickle 

import torch
import torch.nn as nn
from torch.autograd import Variable

class DataReader():

    def initialize(self, filename):

        self.cleaned = self.read_data(filename)
        self.word2idx, self.idx2word = self.make_dictionary(self.cleaned, initial_words=['<EOS>'])
        self.n_letters = len(self.word2idx)

    def read_data(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        pattern = re.compile(r'[^가-힣0-9]+')
        stripped_lines = [pattern.sub('', line).strip() for line in lines]
        cleaned = [line for line in stripped_lines if len(line) > 3 and len(line) < 100]
        return cleaned

    # Random item from a list
    @staticmethod
    def randomChoice(l):
        r = random.randint(0, len(l) - 1)
    #     print(r)
        return l[r]

    # Get a random category and random line from that category
    def randomTrainingPair(self):
        line = self.randomChoice(self.cleaned)
        return line

    # One-hot matrix of first to last letters (not including EOS) for input
    def inputTensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][self.word2idx[letter]] = 1
        return tensor

    # LongTensor of second letter to end (EOS) for target
    def targetTensor(self, line):
        letter_indexes = [self.word2idx[line[li]] for li in range(1, len(line))]
        letter_indexes.append(0) # EOS
        return torch.LongTensor(letter_indexes)

    # Make category, input, and target tensors from a random category, line pair
    def randomTrainingExample(self):
        line = self.randomTrainingPair()
        input_line_tensor = Variable(self.inputTensor(line))
        target_line_tensor = Variable(self.targetTensor(line))
        return input_line_tensor, target_line_tensor

    @staticmethod
    def make_dictionary(sentences, vocabulary_size=None, initial_words=['<SOS>', '<EOS>', '<PAD>', '<UNK>']):
        """sentences : list of lists"""
        
        counter = Counter()
        for words in sentences:
            counter.update(words)
        
        if vocabulary_size is None:
            vocabulary_size = len(counter.keys())
        
        vocab_words = counter.most_common(vocabulary_size - len(initial_words))
        
        for initial_word in reversed(initial_words):
            vocab_words.insert(0, (initial_word, 0))
        
        word2idx = {word:idx for idx, (word, count) in enumerate(vocab_words)}
        idx2word = {idx:word for word, idx in word2idx.items()}
        
        return word2idx, idx2word

    def save(self, filename):
        
        save_dict = {'word2idx': self.word2idx, 
                     'idx2word': self.idx2word, 
                     'n_letters': self.n_letters}

        with open(filename, 'wb') as file:
            pickle.dump(save_dict, file)

    def load(self, filename):

        with open(filename, 'rb') as file:
            save_dict = pickle.load(file)

        self.word2idx = save_dict['word2idx']
        self.idx2word = save_dict['idx2word']
        self.n_letters = save_dict['n_letters']

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


import konlpy
from konlpy.tag import Twitter

class DataReader2():

    def initialize(self, filename):

        self.cleaned = self.read_data(filename)
        self.word2idx, self.idx2word = self.make_dictionary(self.cleaned, vocabulary_size=10000, initial_words=['<EOS>', '<UNK>'])
        self.n_letters = len(self.word2idx)

    def read_data(self, filename):
        tagger = Twitter()
        with open(filename, 'r') as f:
            lines = f.readlines()

        stripped_lines = [line for line in lines][:10000]
        cleaned = [tagger.pos(line) for line in stripped_lines if len(line) > 5 and len(line) < 50]
        return cleaned

    # Random item from a list
    @staticmethod
    def randomChoice(l):
        r = random.randint(0, len(l) - 1)
    #     print(r)
        return l[r]

    # Get a random category and random line from that category
    def randomTrainingPair(self):
        line = self.randomChoice(self.cleaned)
        return line

    # One-hot matrix of first to last letters (not including EOS) for input
    def inputTensor(line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][(self.word2idx[letter] if letter in word2idx else word2idx['<UNK>'])] = 1
        return tensor

    # LongTensor of second letter to end (EOS) for target
    def targetTensor(line):
        letter_indexes = [(self.word2idx[line[li]] if line[li] in word2idx else word2idx['<UNK>']) for li in range(1, len(line))]
        letter_indexes.append(0) # EOS
        return torch.LongTensor(letter_indexes)
    
    # Make category, input, and target tensors from a random category, line pair
    def randomTrainingExample(self):
        line = self.randomTrainingPair()
        input_line_tensor = Variable(self.inputTensor(line))
        target_line_tensor = Variable(self.targetTensor(line))
        return input_line_tensor, target_line_tensor

    @staticmethod
    def make_dictionary(sentences, vocabulary_size=None, initial_words=['<SOS>', '<EOS>', '<PAD>', '<UNK>']):
        """sentences : list of lists"""
        
        counter = Counter()
        for words in sentences:
            counter.update(words)
        
        if vocabulary_size is None:
            vocabulary_size = len(counter.keys())
        
        vocab_words = counter.most_common(vocabulary_size - len(initial_words))
        
        for initial_word in reversed(initial_words):
            vocab_words.insert(0, (initial_word, 0))
        
        word2idx = {word:idx for idx, (word, count) in enumerate(vocab_words)}
        idx2word = {idx:word for word, idx in word2idx.items()}
        
        return word2idx, idx2word

    def save(self, filename):
        
        save_dict = {'word2idx': self.word2idx, 
                     'idx2word': self.idx2word, 
                     'n_letters': self.n_letters}

        with open(filename, 'wb') as file:
            pickle.dump(save_dict, file)

    def load(self, filename):

        with open(filename, 'rb') as file:
            save_dict = pickle.load(file)

        self.word2idx = save_dict['word2idx']
        self.idx2word = save_dict['idx2word']
        self.n_letters = save_dict['n_letters']