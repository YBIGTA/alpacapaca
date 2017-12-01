import re
import random
from collections import Counter
import time
import math
import pickle 

import torch
import torch.nn as nn
from torch.autograd import Variable

class LineGenerator():

    def __init__(self, model, data_reader, max_length=20):

        self.rnn = model
        self.data_reader = data_reader
        self.max_length = max_length

    # Sample from a category and starting letter
    def sample(self, start_letter='A'):
        input = Variable(self.data_reader.inputTensor(start_letter))
        hidden = self.rnn.initHidden()

        output_name = start_letter

        for i in range(self.max_length):
            output, hidden = self.rnn(input[0], hidden)
            topv, topi = output.data.topk(1)
            topi = topi[0][0]
            if topi == 0:
                break
            else:
                letter = self.data_reader.idx2word[topi]
                output_name += letter
            input = Variable(self.data_reader.inputTensor(letter))

        return output_name

    # Get multiple samples from one category and multiple starting letters
    def samples(self, start_letters='ABC'):
        results = []
        for start_letter in start_letters:
            results.append(self.sample(start_letter))
        return results