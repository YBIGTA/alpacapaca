import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

class BasicLSTM(nn.Module):
    def __init__(self, input_size, rnn_size, hidden_size, target_size, use_gpu=False):
        super(BasicLSTM, self).__init__()
        
        self.rnn_size = rnn_size
        self.i2h = nn.Linear(input_size, hidden_size)
        
        self.lstm = nn.LSTM(hidden_size, rnn_size, batch_first=True)
        
        self.hidden2out = nn.Linear(rnn_size, target_size)
        
        self.use_gpu = use_gpu
        
        # self.softmax = nn.LogSoftmax()

    def forward(self, inputs):
        # inputs : (batch_size, seq_size, embedding_size)
        embeds = self.i2h(inputs)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        self.hidden = self.hidden[0].cuda(), self.hidden[1].cuda()
        # print(lstm_out.size())
        output_space = self.hidden2out(lstm_out)
        # output_scores = self.softmax(output_space)
        return output_space

    def init_hidden(self, batch_size):
        if self.use_gpu:
            self.hidden = (Variable(torch.zeros(1, batch_size, self.rnn_size).cuda()),
                Variable(torch.zeros(1, batch_size, self.rnn_size).cuda()))
        else:
            self.hidden = (Variable(torch.zeros(1, batch_size, self.rnn_size)),
                Variable(torch.zeros(1, batch_size, self.rnn_size)))
            
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
        # batch norm
        
        self.highway_t = nn.Linear(hidden_size, hidden_size)
        self.highway_g = nn.Linear(hidden_size, hidden_size)
        
        self.dense1 = nn.Linear(hidden_size, hidden_size//2)
        self.dense2 = nn.Linear(hidden_size//2, wordvec_size)
        
    def forward(self, inputs):
        # inputs : (batch_size, seq_size, embedding_size)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        self.last_out = lstm_out
        self.relu_out = self.relu(self.hidden2out_1(self.last_out))
        self.tanh_out = self.tanh(self.hidden2out_2(self.last_out))
        self.concat = torch.cat([self.relu_out, self.tanh_out], dim=2)
        
        def _highway(in_features, num_layers=1, bias=-2.0):
            t = self.sigmoid(self.highway_t(in_features) + bias)
            g = nn.functional.relu(self.highway_g(in_features))
            return t * g + (1-t) * in_features
        
        self.highway_out = _highway(self.concat)
        return self.dense2(self.dense1(self.highway_out))

    def init_hidden(self, batch_size, use_gpu=False):
        hidden_shape = (self.num_layers, batch_size, self.hidden_size)
        if use_gpu:
            self.hidden = (Variable(torch.zeros(hidden_shape).cuda()),
                Variable(torch.zeros(hidden_shape).cuda()))
        else:
            self.hidden = (Variable(torch.zeros(hidden_shape)),
                Variable(torch.zeros(hidden_shape)))