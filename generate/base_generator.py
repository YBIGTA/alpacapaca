import re
import random
from collections import Counter
import time
import math
import pickle 
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

PAD_TOKEN = '<Pad>/Pad'

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

class LineSampleGenerator():

    def __init__(self, model, data_reader, max_length=20):

        self.rnn = model
        self.data_reader = data_reader
        self.max_length = max_length

    # Sample from a category and starting letter
    def sample(self, start_letter='A'):
        start_word = random.choice([key for key in self.data_reader.word2idx.keys() if key[0][0] == start_letter])
        input = Variable(self.data_reader.inputTensor(start_letter))
        hidden = self.rnn.initHidden()

        output_name = start_word[0]

        for i in range(self.max_length):
            output, hidden = self.rnn(input[0], hidden)
            # topv, topi = output.data.topk(1)
            # topi = topi[0][0]
            tt = output.data.topk(10)
            topi = tt[1].numpy()[0][np.random.randint(len(tt[0][0]))]
            if topi == 0:
                break
            else:
                letter = self.data_reader.idx2word[topi]
                output_name = output_name + ' ' + letter[0]
            input = Variable(self.data_reader.inputTensor(letter))

        return output_name

    # Get multiple samples from one category and multiple starting letters
    def samples(self, start_letters='ABC'):
        results = []
        for start_letter in start_letters:
            results.append(self.sample(start_letter))
        return results


class LineEmbeddingGenerator():

    def __init__(self, model, embedding, max_length=20, topn=10, use_gpu=False):

        self.rnn = model
        self.embedding = embedding
        self.max_length = max_length
        self.topn = topn
        self.use_gpu = use_gpu

    def fit_shape(self, vector):
        if self.use_gpu:
            return Variable(torch.FloatTensor(vector.reshape(1, 1, -1))).cuda()
        else:
            return Variable(torch.FloatTensor(vector.reshape(1, 1, -1)))
    
    def _pick_start_word(self, start_letter):
        proper_start_words = [key for key in self.embedding.vocab if key[0] == start_letter and len(key.split('/')[0]) > 1]
        total_count = sum(self.embedding.model.wv.vocab[word].count for word in proper_start_words)
        word_prob = [(word, self.embedding.model.wv.vocab[word].count/total_count) for word in proper_start_words]
        smooth_word_prob = [(word, prob**1/4)for word, prob in word_prob]
        multinomial_pick = np.random.multinomial(1, [prob for word, prob in smooth_word_prob])
        pick_index = multinomial_pick.argsort()[-1]
        pick_word = word_prob[pick_index][0]
        return pick_word
    
    def sample(self, start_letter='가'):
        start_word = self._pick_start_word(start_letter)
        self.input_vector = self.embedding.vectorize(start_word)
        self.input_tensor = self.fit_shape(self.input_vector)

        self.rnn.init_hidden(1, self.use_gpu)

        self.output_line = start_word.split('/')[0]

        for i in range(self.max_length):
            self.output_tensor = self.rnn(self.input_tensor)
            # topv, topi = self.output_scores.data.topk(1)
            # topi = topi[0][0]
            self.output_vector = self.output_tensor.cpu().data[0, 0, :].numpy()
            topn = self.embedding.model.wv.similar_by_vector(self.output_vector, topn=self.topn)
            next_word = random.choice(topn)[0]
            if next_word == PAD_TOKEN:
                break
            self.output_line = self.output_line + ' ' + next_word.split('/')[0]
            self.input_vector = self.embedding.vectorize(next_word)
            self.input_tensor = self.fit_shape(self.input_vector)

        return self.output_line

    def samples(self, start_letters='가나다'):
        results = []
        for start_letter in start_letters:
            results.append(self.sample(start_letter))
        return results
    
class OneLineGenerator():

    def __init__(self, model, embedding, max_length=100, topn=100, use_gpu=False):

        self.rnn = model
        self.embedding = embedding
        self.max_length = max_length
        self.topn = topn
        self.use_gpu = use_gpu

    def fit_shape(self, vector):
        if self.use_gpu:
            return Variable(torch.FloatTensor(vector.reshape(1, 1, -1))).cuda()
        else:
            return Variable(torch.FloatTensor(vector.reshape(1, 1, -1)))
    
    def _pick_start_word(self, start_letter):
        proper_start_words = [key for key in self.embedding.vocab if key[0] == start_letter and len(key.split('/')[0]) > 1]
        total_count = sum(self.embedding.model.wv.vocab[word].count for word in proper_start_words)
        word_prob = [(word, self.embedding.model.wv.vocab[word].count/total_count) for word in proper_start_words]
        smooth_word_prob = [(word, prob**1/4)for word, prob in word_prob]
        multinomial_pick = np.random.multinomial(1, [prob for word, prob in smooth_word_prob])
        pick_index = multinomial_pick.argsort()[-1]
        pick_word = word_prob[pick_index][0]
        return pick_word

    def samples(self, start_letters='가나다'):
        start_letter = start_letters[0]
        start_word = self._pick_start_word(start_letter)
        self.input_vector = self.embedding.vectorize(start_word)
        self.input_tensor = self.fit_shape(self.input_vector)
        self.rnn.init_hidden(1, self.use_gpu)
        
        results = []
        self.output_line = start_word.split('/')[0]
        
        target_index = 1

        for i in range(self.max_length):
#             print(self.output_line, target_index)
            target_letter = start_letters[target_index] if target_index < 3 else None
            
            self.output_tensor = self.rnn(self.input_tensor)
            self.output_vector = self.output_tensor.cpu().data[0, 0, :].numpy()
            topn = self.embedding.model.wv.similar_by_vector(self.output_vector, topn=self.topn)
            target_outputs = [word for word, similarity in topn if word[0] == target_letter]
            if target_outputs:
                next_word = random.choice(target_outputs)
                target_index += 1
                results.append(self.output_line)
                self.output_line = next_word.split('/')[0]
            else:
                next_word = random.choice(topn)[0]
                self.output_line = self.output_line + ' ' + next_word.split('/')[0]
            if target_index > 2 and next_word == PAD_TOKEN:
                results.append(self.output_line)
                break
            self.input_vector = self.embedding.vectorize(next_word)
            self.input_tensor = self.fit_shape(self.input_vector)
        else:
            results.append(self.output_line)
        
        if target_index > 2:
            print('Success')
        else:
            print('Fail')
            
        return results