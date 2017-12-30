from flask import Flask
from flask_restplus import Resource, Api
from flask_cors import CORS
from flask import abort

import os, sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
print(parentdir)
sys.path.insert(0, parentdir) 

import torch, gensim
import numpy as np
from data_load.embedding_loader import EmbeddingDataset, UniqueEmbeddingDataset, EmbeddingDataLoader
from model.rnn import DeepLSTM
from train.base_trainer import TrainValTrainer
from generate.base_generator import LineEmbeddingGenerator
from embedding.word2vec_model import ReplacingWord2VecModel

import torch
import torch.nn as nn
from torch.autograd import Variable

class Highway(nn.Module):
    def __init__(self, num_features):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(num_features, num_features)
        self.gate_layer = nn.Linear(num_features, num_features)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        
        normal_fc = self.tanh(self.fc1(inputs))
        transformation_layer = self.sigmoid(self.gate_layer(inputs))
        
        carry_layer = 1 - transformation_layer
        info_flow = normal_fc * transformation_layer + (1-transformation_layer)*inputs
        return info_flow

class DeepLSTM(nn.Module):
    def __init__(self, wordvec_size=300, hidden_size=1024, linear_size=512,
                 num_layers=3,
                 highway_num_layers=2, bidirectional=False):
        super(DeepLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=wordvec_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=bidirectional,
                            dropout=0.5, batch_first=True)
        
        self.hidden2out_1 = nn.Linear(hidden_size, hidden_size//2)
        self.hidden2out_2 = nn.Linear(hidden_size, hidden_size//2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.batchnorm1 = nn.BatchNorm1d(hidden_size//2)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size//2)
        self.batchnorm3 = nn.BatchNorm1d(hidden_size//2)
        
        self.highway = nn.ModuleList([Highway(hidden_size) for _ in range(highway_num_layers)])
        
        self.dense1 = nn.Linear(hidden_size, hidden_size//2)
        self.dense2 = nn.Linear(hidden_size//2, wordvec_size)
        
    def forward(self, inputs):
        # inputs : (batch_size, seq_size, embedding_size)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        self.last_out = lstm_out
        self.relu_out = self.batchnorm1(self.relu(self.hidden2out_1(self.last_out)).view(-1, 512, inputs.size(1))).view(-1, inputs.size(1), 512)
        self.tanh_out = self.batchnorm2(self.tanh(self.hidden2out_2(self.last_out)).view(-1, 512, inputs.size(1))).view(-1, inputs.size(1), 512)
        self.concat = torch.cat([self.relu_out, self.tanh_out], dim=2)
        
        x = self.concat
        for current_highway in self.highway:
            x = current_highway(x)
            
        return self.dense2(self.batchnorm3(self.relu(self.dense1(x)).view(-1, 512, inputs.size(1))).view(-1, inputs.size(1), 512))

    def init_hidden(self, batch_size, use_gpu=False):
        hidden_shape = (self.num_layers, batch_size, self.hidden_size)
        if use_gpu:
            self.hidden = (Variable(torch.zeros(hidden_shape).cuda()),
                Variable(torch.zeros(hidden_shape).cuda()))
        else:
            self.hidden = (Variable(torch.zeros(hidden_shape)),
                Variable(torch.zeros(hidden_shape)))

embedding = ReplacingWord2VecModel.load_from('only_lyrics_no_japanese_word2vec_100', output_dirpath='/home/ubuntu/alpacapaca/embedding/output/')
model = torch.load('/home/ubuntu/alpacapaca/train/output/80_first_success.pth', map_location=lambda storage, loc: storage)
model.eval()
model = model.cpu()

generator = LineEmbeddingGenerator(model=model, embedding=embedding, max_length=20, topn=5, use_gpu=False)

app = Flask(__name__)
CORS(app)
api = Api(app)

@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world, 안녕!'}

@api.route('/alpaca/<string:input_word>')
class Alpacapaca(Resource):
    
    def get(self, input_word):
        
        try:
            results = generator.samples(input_word)

            return {'success': True,
                    'results': results}
        except:
            abort(500)

if __name__ == '__main__':    
    print(generator.samples('김연아'))
    
    app.run(host='0.0.0.0', debug=True)