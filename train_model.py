
# coding: utf-8

# In[ ]:


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


# In[ ]:


class LSTM(nn.Module):
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

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)))

    def forward(self, sentence_inp, mask):
        # sentence_inp - batch_size x truncate_length x num_features
        # mask - batch_size x truncate_length
        batch_size = sentence_inp.size()[0]
        self.hidden = self.init_hidden(batch_size)
        # lstm expects batch_size x truncate_length x num_features because of batch_first=True
        outputs, self.hidden = self.lstm(sentence_inp)
        out_masked = torch.mul(outputs, mask.unsqueeze(2).expand_as(outputs))
        out_masked_avg = torch.div(out_masked.sum(dim=1), 
                                   mask.sum(dim=1).unsqueeze(1).expand(batch_size, self.hidden_dim))
        return out_masked_avg


# In[ ]:


class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, truncate_length):
        super(CNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.truncate_length= truncate_length

        self.conv = nn.Conv1d(input_dim, hidden_dim, 3, padding=1)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, sentence_inp):
        # sentence_inp is batch_size x truncate_length x num_features
        x = self.conv(sentence_inp)
        x = self.tanh(x)
        x = self.drop(x)
        # TODO do average pooling over hidden states
        return x


# In[ ]:


# batch_size = BATCH_SIZE
# data_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size,
#                                               shuffle=True, drop_last=True)
# for batch in tqdm(data_loader):
#     q_body = Variable(batch["q_body"], requires_grad=True) # batch_size x truncate_length x 200]
#     cand_bodies = Variable(batch["candidate_bodies"], requires_grad=True) # batch_size x num_cands x truncate_length x 200
#     q_title = Variable(batch["q_title"], requires_grad=True)
#     cand_titles = Variable(batch["candidate_titles"], requires_grad=True)
#     num_cands = cand_titles.size()[1]
#     q_body_mask, q_title_mask = Variable(batch["q_body_mask"]), Variable(batch["q_title_mask"])
#     cand_body_masks, cand_title_masks = Variable(batch["candidate_body_masks"]), Variable(batch["candidate_title_masks"])
#     break


# In[ ]:


def evaluate(all_ranked_labels):
    evaluator = Evaluation(all_ranked_labels)
    MAP = evaluator.MAP()*100
    MRR = evaluator.MRR()*100
    P1 = evaluator.Precision(1)*100
    P5 = evaluator.Precision(5)*100
    return MAP, MRR, P1, P5


# In[ ]:


def run_epoch(dataset, is_training, model, optimizer, batch_size, margin):
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
        q_body_mask = Variable(batch["q_body_mask"], requires_grad=True) # batch_size x truncate_length
        q_title_mask = Variable(batch["q_title_mask"], requires_grad=True)
        cand_body_masks = Variable(batch["candidate_body_masks"], requires_grad=True) # batch_size x num_cands x truncate_length
        cand_title_masks = Variable(batch["candidate_title_masks"], requires_grad=True)
        num_cands = cand_titles.size()[1]
        if is_training:
            optimizer.zero_grad()
        q_body_enc, q_title_enc = model(q_body, q_body_mask), model(q_title, q_title_mask) # output is batch_size  x enc_length
        cand_body_encs = model(cand_bodies.view(batch_size*num_cands, TRUNCATE_LENGTH, 200), # output is (batch_size x num_cands) x enc_length
                               cand_body_masks.view(batch_size*num_cands, TRUNCATE_LENGTH)) 
        cand_title_encs = model(cand_titles.view(batch_size*num_cands, TRUNCATE_LENGTH, 200),
                                cand_title_masks.view(batch_size*num_cands, TRUNCATE_LENGTH))
        q_enc = q_title_enc + q_body_enc / 2.0
        candidate_encs = cand_title_encs + cand_body_encs / 2.0
        #domain_predictions = domain_classifier(q_enc, candidate_ends)
        #loss(domain_predictions, target_predictions)
        #domain_optimizer.step()
        candidate_encs = candidate_encs.view(batch_size, num_cands, -1) # batch_size x num_cands x enc_length
        query_encs = q_enc.view(batch_size, 1, -1).repeat(1, num_cands, 1) # batch_size x (num_cands) x enc_length
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-08)(candidate_encs, query_encs) # batch_size x (num_cands)
        
        if is_training:
            target = Variable(torch.zeros(batch_size).long(), requires_grad=True)
            loss = torch.nn.MultiMarginLoss(margin=margin)(cos, target)
            #total_loss = loss - domain_loss
            #total_loss.backward()
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.cpu().data[0])
        else:
            # do evaluation stuff
            sorted_cos, ind = cos.sort(1, descending=True)
            labels = batch["labels"]
            for i in range(batch_size): 
                all_ranked_labels.append(labels[i][ind.data[i]])
    if is_training:
        avg_loss = np.mean(losses)
        return avg_loss
    else:
        return evaluate(all_ranked_labels)
    


# In[ ]:


def train_model(train_data, dev_data, test_data, model, batch_size=50, margin=1, num_epochs=50, lr=1.0, weight_decay=0):
    print("start train_model")
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters())
    result_table = PrettyTable(["Epoch", "train loss", "dev MAP", "dev MRR", "dev P@1", "dev P@5"] +
                                    ["tst MAP", "tst MRR", "tst P@1", "tst P@5"])
    for epoch in range(1, num_epochs + 1):
        print("epoch", epoch)
        train_loss = run_epoch(train_data, True, model, optimizer, batch_size, margin)
        dev_MAP, dev_MRR, dev_P1, dev_P5 = run_epoch(dev_data, False, model, optimizer, batch_size, margin)
        test_MAP, test_MRR, test_P1, test_P5 = run_epoch(test_data, False, model, optimizer, batch_size, margin)
        result_table.add_row(
                            [ epoch ] +
                            [ "%.2f" % x for x in [train_loss] + [ dev_MAP, dev_MRR, dev_P1, dev_P5 ] +
                                        [ test_MAP, test_MRR, test_P1, test_P5 ] ])
        print("{}".format(result_table))


# In[ ]:


TEXT_TOKENIZED_FILE = 'askubuntu/text_tokenized.txt'
TRAIN_FILE = 'askubuntu/train_random.txt'
DEV_FILE = 'askubuntu/dev.txt'
TEST_FILE = 'askubuntu/test.txt'

TRUNCATE_LENGTH = 100


# In[ ]:


train_dataset = TrainQuestionDataset(TEXT_TOKENIZED_FILE, TRAIN_FILE, truncate=TRUNCATE_LENGTH, test_subset=9)
dev_dataset = EvalQuestionDataset(train_dataset.id_to_question, DEV_FILE, truncate=TRUNCATE_LENGTH, test_subset=9)
test_dataset = EvalQuestionDataset(train_dataset.id_to_question, TEST_FILE, truncate=TRUNCATE_LENGTH, test_subset=9)


# In[ ]:


BATCH_SIZE = 3
NUM_EPOCHS = 2
MARGIN = 0.5
model = LSTM(200, 240, 1, TRUNCATE_LENGTH)


# In[ ]:


train_model(train_dataset, dev_dataset, test_dataset, model, num_epochs=NUM_EPOCHS, margin=MARGIN, batch_size=BATCH_SIZE)

