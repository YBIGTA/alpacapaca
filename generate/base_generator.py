import re
import random
from collections import Counter
import time
import math
import pickle 
import numpy as np
import hgtk

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
        
    def change_multiple(self, start_letter):
    #     한자음 '녀, 뇨, 뉴, 니, 랴, 려, 례, 료, 류, 리' 등 ㄴ 또는 ㄹ+ㅣ나 ㅣ로 시작하는 이중모음이 단어 첫머리에 올 때 '여, 요, 유, 이', '야, 여, 예, 요, 유, 이'로 발음한다. 
    #     한자음 '라, 래, 로, 뢰, 루, 르' 등 ㄹ+ㅣ를 제외한 단모음이 단어 첫머리에 올 때 '나, 내, 노, 뇌, 누, 느'로 발음한다.
        zaeum_list = ['ㄴ', 'ㄹ']
        single_moeum_list = ['ㅔ', 'ㅐ', 'ㅟ', 'ㅚ', 'ㅡ', 'ㅓ', 'ㅏ', 'ㅜ', 'ㅗ']
        double_moeum_list = ['ㅣ', 'ㅑ', 'ㅕ', 'ㅛ', 'ㅠ', 'ㅖ', 'ㅒ']

        decompose = hgtk.letter.decompose(start_letter)
        first_zaeum = decompose[0]
        first_moeum = decompose[1]
        if first_zaeum in zaeum_list:
            #first case
            if first_moeum in double_moeum_list:
                new_letter_component = list(decompose)
                new_letter_component[0] = 'ㅇ'
                new_letter = hgtk.letter.compose(*new_letter_component)
                return [start_letter, new_letter]
            elif first_zaeum == 'ㄹ' and first_moeum in single_moeum_list:
                new_letter_component = list(decompose)
                new_letter_component[0] = 'ㄴ'
                new_letter = hgtk.letter.compose(*new_letter_component)
                return [start_letter, new_letter]
            else:
                return [start_letter]
        else:
            return [start_letter]
    
    def _pick_start_word(self, start_letter):
        word_list = self.change_multiple(start_letter) #두음법칙 적용해야 할경우 고르자

        if len(word_list) == 1: #word_list 길이가 1일때 - 두음법칙 적용 안함
            proper_start_words = [key for key in self.embedding.vocab if key[0] == start_letter and len(key.split('/')[0]) > 1]
        else: #word_list 길이가 2일때 - 두음법칙 적용 함
            original_letter = word_list[0]
            new_letter = word_list[1]

            proper_start_words_1 = [key for key in self.embedding.vocab if key[0] == original_letter and len(key.split('/')[0]) > 1]
            proper_start_words_2 = [key for key in self.embedding.vocab if key[0] == new_letter and len(key.split('/')[0]) > 1]
            proper_start_words = proper_start_words_1 + proper_start_words_2
            
        total_count = sum(self.embedding.model.wv.vocab[word].count for word in proper_start_words)
        word_prob = [(word, self.embedding.model.wv.vocab[word].count/total_count) for word in proper_start_words]
#         print(word_prob)
#         print(word_prob)
#         smooth_word_prob = [(word, prob) for word, prob in word_prob]
        multinomial_pick = np.random.multinomial(1, [prob for word, prob in word_prob])
        pick_index = multinomial_pick.argsort()[-1]
        pick_word = word_prob[pick_index][0]
        return pick_word
    
    def sample(self, start_letter, raw):
        start_word = self._pick_start_word(start_letter)
        self.input_vector = self.embedding.vectorize(start_word)
        self.input_tensor = self.fit_shape(self.input_vector)

        self.rnn.init_hidden(1, self.use_gpu)
        
        if not raw:
            self.output_line = start_word.split('/')[0]
        else:
            self.output_line = start_word

        for i in range(self.max_length):
            self.output_tensor = self.rnn(self.input_tensor)
            # topv, topi = self.output_scores.data.topk(1)
            # topi = topi[0][0]
            self.output_vector = self.output_tensor.cpu().data[0, 0, :].numpy()
            topn = self.embedding.model.wv.similar_by_vector(self.output_vector, topn=self.topn)

            topn = [(word, prob) for word, prob in topn if word.split('/')[0] not in ['릅쯔쯔르', '요룰', '일랜시아', '쿠탓테', '데키루코토', '히토츠즈츠', '골게', '일백구십사', '반야바라밀다심경', ]]
#             print(topn)
            next_word = random.choice(topn)[0]
            if next_word == PAD_TOKEN:
                break
#             print(next_word)
            if not raw:
                self.output_line = self.output_line + ' ' + next_word.split('/')[0]
            else:
                self.output_line = self.output_line + ' ' + next_word
            self.input_vector = self.embedding.vectorize(next_word)
            self.input_tensor = self.fit_shape(self.input_vector)

        return self.output_line

    def samples(self, start_letters='가나다', raw=False):
        results = []
        for start_letter in start_letters:
            results.append(self.sample(start_letter, raw=raw))
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
        smooth_word_prob = [(word, prob**1/4) for word, prob in word_prob]
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