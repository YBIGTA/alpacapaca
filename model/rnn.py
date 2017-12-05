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
    def __init__(self, batch_size, input_size, rnn_size, hidden_size, target_size):
        super(BasicLSTM, self).__init__()
        
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.i2h = nn.Linear(input_size, hidden_size)
        
        self.lstm = nn.LSTM(hidden_size, rnn_size, batch_first=True)
        
        self.hidden2out = nn.Linear(rnn_size, target_size)
        self.hidden = self.init_hidden()
        
        self.softmax = nn.LogSoftmax()

    def forward(self, inputs):
        # inputs : (batch_size, seq_size, embedding_size)
        
        embeds = self.i2h(inputs)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # print(lstm_out.size())
        output_space = self.hidden2out(lstm_out)
        output_scores = self.softmax(output_space)
        return output_scores

    def init_hidden(self):
        return (Variable(torch.zeros(1, self.batch_size, self.rnn_size)),
                Variable(torch.zeros(1, self.batch_size, self.rnn_size)))