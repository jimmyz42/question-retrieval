
# coding: utf-8

# In[1]:


import read_input
from read_input import TrainQuestionDataset, EvalQuestionDataset
from prettytable import PrettyTable

from evaluation import Evaluation
import torch
from torch.autograd import Variable
import torch.utils
import torch.utils.data
from tqdm import tqdm
from torch import nn
import numpy as np


# In[2]:


class LSTM(nn.Module):
    # todo: deal with padding
    # todo: fix parameters, structure, etc
    def __init__(self, input_dim, hidden_dim, num_layers, truncate_length):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.truncate_length= truncate_length
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        #self.hidden = self.init_hidden()

#     def init_hidden(self):
#         # Before we've done anything, we dont have any hidden state.
#         # Refer to the Pytorch documentation to see exactly
#         # why they have this dimensionality.
#         # The axes semantics are (num_layers, minibatch_size, hidden_dim)
#         return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)),
#                 Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)))

    def forward(self, sentence_inp):
        # sentence_inp is batch_size x truncate_length x num_features
        batch_size = sentence_inp.size()[0]
        #self.hidden = self.init_hidden() # do we need this?
        # lstm expects batch_size x truncate_length x num_features
        lstm_out, self.hidden = self.lstm(sentence_inp)
        encoding = self.hidden[0].view(batch_size, self.hidden_dim)
        return encoding


# In[3]:


# batch_size = BATCH_SIZE
# data_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size,
#                                               shuffle=True, drop_last=True)
# for batch in tqdm(data_loader):
#     q_body = Variable(batch["q_body"], requires_grad=True) # batch_size x truncate_length x 200]
#     cand_bodies = Variable(batch["candidate_bodies"], requires_grad=True) # batch_size x num_cands x truncate_length x 200
#     q_title = Variable(batch["q_title"], requires_grad=True)
#     cand_titles = Variable(batch["candidate_titles"], requires_grad=True)
#     num_cands = cand_titles.size()[1]
#     q_body_enc, q_title_enc = model(q_body), model(q_title) # batch_size  x enc_length
#     cand_body_encs = model(cand_bodies.view(batch_size*num_cands, TRUNCATE_LENGTH, 200)) # (batch_size x num_cands) x enc_length
#     cand_title_encs = model(cand_titles.view(batch_size*num_cands, TRUNCATE_LENGTH, 200))
#     q_enc = q_title_enc + q_body_enc / 2.0
#     candidate_encs = cand_title_encs + cand_body_encs / 2.0
#     candidate_encs = candidate_encs.view(batch_size, num_cands, -1) # batch_size x num_cands x enc_length
#     query_encs = q_enc.view(batch_size, 1, -1).repeat(1, num_cands, 1) # batch_size x (num_cands) x enc_length
#     cos = torch.nn.CosineSimilarity(dim=2, eps=1e-08)(candidate_encs, query_encs) # batch_size x (num_cands)
#     break


# In[13]:


def evaluate(all_ranked_labels):
    evaluator = Evaluation(all_ranked_labels)
    MAP = evaluator.MAP()*100
    MRR = evaluator.MRR()*100
    P1 = evaluator.Precision(1)*100
    P5 = evaluator.Precision(5)*100
    return MAP, MRR, P1, P5


# In[11]:


def run_epoch(dataset, is_training, model, optimizer, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, drop_last=True)
    losses = []
    all_ranked_labels = []
    if is_training:
        model.train()
    else:
        model.eval()

    for batch in tqdm(data_loader):
        q_body = Variable(batch["q_body"], requires_grad=True) # batch_size x truncate_length x 200]
        cand_bodies = Variable(batch["candidate_bodies"], requires_grad=True) # batch_size x num_cands x truncate_length x 200
        q_title = Variable(batch["q_title"], requires_grad=True)
        cand_titles = Variable(batch["candidate_titles"], requires_grad=True)
        num_cands = cand_titles.size()[1]
        if is_training:
            optimizer.zero_grad()
        q_body_enc, q_title_enc = model(q_body), model(q_title) # batch_size  x enc_length
        cand_body_encs = model(cand_bodies.view(batch_size*num_cands, TRUNCATE_LENGTH, 200)) # (batch_size x num_cands) x enc_length
        cand_title_encs = model(cand_titles.view(batch_size*num_cands, TRUNCATE_LENGTH, 200))
        q_enc = q_title_enc + q_body_enc / 2.0
        candidate_encs = cand_title_encs + cand_body_encs / 2.0
        candidate_encs = candidate_encs.view(batch_size, num_cands, -1) # batch_size x num_cands x enc_length
        query_encs = q_enc.view(batch_size, 1, -1).repeat(1, num_cands, 1) # batch_size x (num_cands) x enc_length
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-08)(candidate_encs, query_encs) # batch_size x (num_cands)
        
        if is_training:
            target = Variable(torch.zeros(batch_size).long(), requires_grad=True)
            loss = torch.nn.MultiMarginLoss()(cos, target)
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.cpu().data[0])
        else:
            # do evaluation stuff
            sorted_cos, ind = cos.sort()
            labels = batch["labels"]
            for i in range(batch_size): 
                all_ranked_labels.append(labels[i][ind.data[i]])
    if is_training:
        avg_loss = np.mean(losses)
        return avg_loss
    else:
        return evaluate(all_ranked_labels)
    


# In[14]:


def train_model(train_data, dev_data, test_data, model, batch_size=200, num_epochs=50, lr=1.0, weight_decay=0):
    print("start train_model")
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters())
    result_table = PrettyTable(["Epoch", "train loss", "dev MAP", "dev MRR", "dev P@1", "dev P@5"] +
                                    ["tst MAP", "tst MRR", "tst P@1", "tst P@5"])
    for epoch in range(1, num_epochs + 1):
        print("epoch", epoch)
        train_loss = run_epoch(train_data, True, model, optimizer, batch_size)
        dev_MAP, dev_MRR, dev_P1, dev_P5 = run_epoch(dev_data, False, model, optimizer, batch_size)
        test_MAP, test_MRR, test_P1, test_P5 = run_epoch(test_data, False, model, optimizer, batch_size)
        result_table.add_row(
                            [ epoch ] +
                            [ "%.2f" % x for x in [train_loss] + [ dev_MAP, dev_MRR, dev_P1, dev_P5 ] +
                                        [ test_MAP, test_MRR, test_P1, test_P5 ] ])
        print("{}".format(result_table))


# In[7]:


TEXT_TOKENIZED_FILE = 'askubuntu/text_tokenized.txt'
TRAIN_FILE = 'askubuntu/train_random.txt'
DEV_FILE = 'askubuntu/dev.txt'
TEST_FILE = 'askubuntu/test.txt'

TRUNCATE_LENGTH = 150


# In[8]:


train_dataset = TrainQuestionDataset(TEXT_TOKENIZED_FILE, TRAIN_FILE, truncate=TRUNCATE_LENGTH, test_subset=9)
dev_dataset = EvalQuestionDataset(train_dataset.id_to_question, DEV_FILE, truncate=TRUNCATE_LENGTH, test_subset=9)
test_dataset = EvalQuestionDataset(train_dataset.id_to_question, TEST_FILE, truncate=TRUNCATE_LENGTH, test_subset=9)


# In[9]:


BATCH_SIZE = 3
NUM_EPOCHS = 2
model = LSTM(200, 15, 1, TRUNCATE_LENGTH)


# In[15]:


train_model(train_dataset, dev_dataset, test_dataset, model, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

