
# coding: utf-8

# In[1]:


import read_input
from read_input import QuestionDataset

import torch
from torch.autograd import Variable
import torch.utils
import torch.utils.data
from tqdm import tqdm
from torch import nn
import numpy as np


# In[2]:


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, truncate_length, batch_size):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.truncate_length= truncate_length
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)))

    def forward(self, sentence_inp):
        # sentence_inp is batch_size x num_questions x truncate_length x num_features
        num_questions = sentence_inp.size()[1]
        sentences = sentence_inp.view(-1, self.batch_size, self.truncate_length, self.input_dim)
        encodings = []
        for i, sentence in enumerate(sentences):
            self.hidden = self.init_hidden() # do we need this?
            # lstm expects batch_size x truncate_length x num_features
            lstm_out, self.hidden = self.lstm(sentence)
            encodings.append(self.hidden[0].view(self.batch_size, self.hidden_dim))
        encodings_var = torch.cat(encodings)
        # encodings is batch_size x num_questions x enc_length
        return encodings_var.view(self.batch_size, num_questions, self.hidden_dim)


def run_epoch(dataset, is_training, model, optimizer, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, drop_last=True)
    losses = []
    if is_training:
        model.train()
    else:
        model.eval()

    for batch in tqdm(data_loader):
        q_body = Variable(batch["q_body"]) # batch_size x 1 x truncate_length x 200]
        p_body = Variable(batch["p_body"])
        neg_bodies = Variable(batch["neg_bodies"]) # batch_size x num_negs x truncate_length x 200
        q_title = Variable(batch["q_title"])
        p_title = Variable(batch["p_title"])
        neg_titles = Variable(batch["neg_titles"])
        num_negs = neg_titles.size()[1]
        if is_training:
            optimizer.zero_grad()
        q_body_enc, q_title_enc = model(q_body), model(q_title) # batch_size x 1 x enc_length
        p_body_enc, p_title_enc = model(p_body), model(p_title)
        neg_body_encs, neg_title_encs = model(neg_bodies), model(neg_titles) # batch_size x num_negs x enc_length
        q_enc = q_title_enc + q_body_enc / 2.0
        p_enc = p_title_enc + p_body_enc / 2.0
        neg_encs = neg_title_encs + neg_body_encs / 2.0
        candidate_encs = torch.cat((p_enc, neg_encs), dim=1) #batch_size x (num_negs + 1) x enc_length
        query_encs = q_enc.repeat(1, (num_negs+1), 1) # batch_size x (num_negs + 1) x enc_length
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-08)(candidate_encs, query_encs) # batch_size x (num_negs + 1)
        target = Variable(torch.zeros(batch_size).long(), requires_grad=True)
        loss = torch.nn.MultiMarginLoss()(cos, target)

        if is_training:
            loss.backward(retain_graph=True)
            optimizer.step()
        losses.append(loss.cpu().data[0])
    avg_loss = np.mean(losses)
    return avg_loss


# In[5]:


def train_model(train_data, model, batch_size=200, num_epochs=50, lr=1.0, weight_decay=0):
    print("start train_model")
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(1, num_epochs + 1):
        print("epoch", epoch)
        train_loss = run_epoch(train_data, True, model, optimizer, batch_size)
        print("\ntrain_loss", train_loss)



TEXT_TOKENIZED_FILE = 'askubuntu/text_tokenized.txt'
TRAIN_FILE = 'askubuntu/train_random.txt'
DEV_FILE = 'askubuntu/dev.txt'
TEST_FILE = 'askubuntu/test.txt'

TRUNCATE_LENGTH = 150

train_dataset = QuestionDataset(TEXT_TOKENIZED_FILE, TRAIN_FILE, truncate=TRUNCATE_LENGTH, test_subset=9)


BATCH_SIZE = 3
NUM_EPOCHS = 2
model = LSTM(200, 15, 1, TRUNCATE_LENGTH, BATCH_SIZE)

train_model(train_dataset, model, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

