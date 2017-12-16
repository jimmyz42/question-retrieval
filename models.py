
# coding: utf-8

# In[ ]:


from evaluation import Evaluation
import torch
from torch.autograd import Variable
import torch.utils
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import numpy as np


# In[ ]:


class LSTM(nn.Module):
    def __init__(self, embeddings, padding_idx, hidden_dim, num_layers, truncate_length, dropout=0.0, bidirectional=False):
        super(LSTM, self).__init__()
        if bidirectional:
            print('Bidirectinoal LSTM not implemented!!!')
            assert(bidirectional==False)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.truncate_length= truncate_length
        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.weight.requires_grad = False # Freezes the word vectors so we don't train them
        # The LSTM takes word vectors as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        #self.hidden = self.init_hidden()

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)))

    def forward(self, sentence_inp, mask):
        # sentence_inp - batch_size x truncate_length
        # mask - batch_size x truncate_length
        batch_size = sentence_inp.size()[0]
        self.hidden = self.init_hidden(batch_size)
        sentence_vectorized = self.embedding_layer(sentence_inp).float()
        # lstm expects batch_size x truncate_length x num_features because of batch_first=True
        outputs_pre_dropout, self.hidden = self.lstm(sentence_vectorized)
        outputs = self.dropout(outputs_pre_dropout)
        out_masked = torch.mul(outputs, mask.unsqueeze(2).expand_as(outputs))
        out_masked_avg = torch.div(out_masked.sum(dim=1), 
                                   mask.sum(dim=1).unsqueeze(1).expand(batch_size, self.hidden_dim))
        return out_masked_avg


# In[ ]:


class CNN(nn.Module):
    def __init__(self, embeddings, padding_idx, hidden_dim, truncate_length, dropout=0.0):
        super(CNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.truncate_length= truncate_length
        
        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.weight.requires_grad = False # Freezes the word vectors so we don't train them

        self.conv = nn.Conv1d(embed_dim, hidden_dim, 3, padding=1)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, sentence_inp, mask):
        # sentence_inp - batch_size x truncate_length
        # mask - batch_size x truncate_length
        batch_size = sentence_inp.size()[0]
        outputs = self.embedding_layer(sentence_inp).float()
        # batch_size x truncate_length x embedding_dim
        outputs = outputs.transpose(1, 2)
        # outputs needs to be batch_size x embedding_dim x truncate_length (hence the transpose)
        outputs = self.conv(outputs)
        outputs = F.tanh(outputs)
        outputs = self.drop(outputs)
        outputs = outputs.transpose(1, 2)
        # tranpose back so mask works properly
        out_masked = torch.mul(outputs, mask.unsqueeze(2).expand_as(outputs))
        out_masked_avg = torch.div(out_masked.sum(dim=1), 
                                   mask.sum(dim=1).unsqueeze(1).expand(batch_size, self.hidden_dim))
        return out_masked_avg


# In[ ]:


class DomainClassifier(nn.Module):
    # for us hidden_dim1 = 300, hidden_dim2 = 150
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(DomainClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[ ]:


def evaluate(all_ranked_labels):
    evaluator = Evaluation(all_ranked_labels)
    MAP = evaluator.MAP()*100
    MRR = evaluator.MRR()*100
    P1 = evaluator.Precision(1)*100
    P5 = evaluator.Precision(5)*100
    return MAP, MRR, P1, P5

