
# coding: utf-8

# In[1]:


import read_input
from read_input import TrainQuestionDataset, EvalQuestionDataset, embeddings, padding_idx
from prettytable import PrettyTable

from evaluation import Evaluation
import torch
from torch.autograd import Variable
import torch.utils
import torch.utils.data
from tqdm import tqdm
from torch import nn
import numpy as np
import os
from models import LSTM, CNN, evaluate, DomainClassifier


# In[2]:


def run_epoch(dataset, is_training, model, optimizer, batch_size, margin, save_path):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, drop_last=True)
    losses = []
    all_ranked_labels = []
    if is_training:
        model.train()
    else:
        model.eval()
    requires_grad = False
    for batch in tqdm(data_loader):
        q_body = Variable(batch["q_body"], requires_grad=requires_grad) # batch_size x truncate_length x 200]
        cand_bodies = Variable(batch["candidate_bodies"], requires_grad=requires_grad) # batch_size x num_cands x truncate_length x 200
        q_title = Variable(batch["q_title"], requires_grad=requires_grad)
        cand_titles = Variable(batch["candidate_titles"], requires_grad=requires_grad)
        q_body_mask = Variable(batch["q_body_mask"], requires_grad=requires_grad) # batch_size x truncate_length
        q_title_mask = Variable(batch["q_title_mask"], requires_grad=requires_grad)
        cand_body_masks = Variable(batch["candidate_body_masks"], requires_grad=requires_grad) # batch_size x num_cands x truncate_length
        cand_title_masks = Variable(batch["candidate_title_masks"], requires_grad=requires_grad)
        num_cands = cand_titles.size()[1]
        if is_training:
            optimizer.zero_grad()
        q_body_enc, q_title_enc = model(q_body, q_body_mask), model(q_title, q_title_mask) # output is batch_size  x enc_length
        cand_body_encs = model(cand_bodies.view(batch_size*num_cands, TRUNCATE_LENGTH), # output is (batch_size x num_cands) x enc_length
                               cand_body_masks.view(batch_size*num_cands, TRUNCATE_LENGTH)) 
        cand_title_encs = model(cand_titles.view(batch_size*num_cands, TRUNCATE_LENGTH),
                                cand_title_masks.view(batch_size*num_cands, TRUNCATE_LENGTH))
        q_enc = q_title_enc + q_body_enc / 2.0
        candidate_encs = cand_title_encs + cand_body_encs / 2.0
        enc_length = q_enc.size()[-1]
        #domain_predictions = domain_classifier(q_enc, candidate_ends)
        #loss(domain_predictions, target_predictions)
        #domain_optimizer.step()
        candidate_encs = candidate_encs.view(batch_size, num_cands, -1) # batch_size x num_cands x enc_length
        query_encs = q_enc.view(batch_size, 1, -1).expand_as(candidate_encs) # batch_size x (num_cands) x enc_length
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-08)(candidate_encs, query_encs) # batch_size x (num_cands)
        
        if is_training:
            target = Variable(torch.zeros(batch_size).long(), requires_grad=True)
            loss = torch.nn.MultiMarginLoss(margin=margin)(cos, target)
            #total_loss = loss - domain_loss
            #total_loss.backward()
            loss.backward(retain_graph=False)
            optimizer.step()
            losses.append(loss.cpu().data[0])
        else:
            # do evaluation stuff
            sorted_cos, ind = cos.sort(1, descending=True)
            labels = batch["labels"]
            for i in range(batch_size): 
                all_ranked_labels.append(labels[i][ind.data[i]])
    if is_training:
        # save the model
        torch.save(model.state_dict(), save_path)
        avg_loss = np.mean(losses)
        return avg_loss
    else:
        return evaluate(all_ranked_labels)
    


# In[3]:


def train_model(train_data, dev_data, test_data, model, save_dir=None, batch_size=50, margin=1, num_epochs=50, lr=1.0, weight_decay=0):
    if (save_dir is not None) and (not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    print("start train_model")
    print("****************************************")
    print("Batch size: {}, margin: {}, num_epochs: {}, lr: {}".format(batch_size, margin, num_epochs, lr))
    print("Model", model)
    print("*****************************************")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    result_table = PrettyTable(["Epoch", "train loss", "dev MAP", "dev MRR", "dev P@1", "dev P@5"] +
                                    ["tst MAP", "tst MRR", "tst P@1", "tst P@5"])
    for epoch in range(1, num_epochs + 1):
        print("epoch", epoch)
        if save_dir is None:
            save_path = None
        else:
            save_path = os.path.join(save_dir, 'epoch{}.pkl'.format(epoch))
        train_loss = run_epoch(train_data, True, model, optimizer, batch_size, margin, save_path)
        dev_MAP, dev_MRR, dev_P1, dev_P5 = run_epoch(dev_data, False, model, optimizer, batch_size, margin, save_path)
        test_MAP, test_MRR, test_P1, test_P5 = run_epoch(test_data, False, model, optimizer, batch_size, margin, save_path)
        result_table.add_row(
                            [ epoch ] +
                            [ "%.2f" % x for x in [train_loss] + [ dev_MAP, dev_MRR, dev_P1, dev_P5 ] +
                                        [ test_MAP, test_MRR, test_P1, test_P5 ] ])
        print("{}".format(result_table))


# In[4]:


TEXT_TOKENIZED_FILE = 'askubuntu/text_tokenized.txt'
TRAIN_FILE = 'askubuntu/train_random.txt'
DEV_FILE = 'askubuntu/dev.txt'
TEST_FILE = 'askubuntu/test.txt'

TRUNCATE_LENGTH = 100


# In[10]:


train_dataset = TrainQuestionDataset(TEXT_TOKENIZED_FILE, TRAIN_FILE, truncate=TRUNCATE_LENGTH, test_subset=None)
dev_dataset = EvalQuestionDataset(train_dataset.id_to_question, DEV_FILE, truncate=TRUNCATE_LENGTH, test_subset=None)
test_dataset = EvalQuestionDataset(train_dataset.id_to_question, TEST_FILE, truncate=TRUNCATE_LENGTH, test_subset=None)


# In[11]:


DROPOUT_PROBS = [0.0, 0.1, 0.2, 0.3] # Taken from paper
DROPOUT = 0.1
BIDIRECTIONAL = False

#model = LSTM(embeddings, padding_idx, 15, 1, TRUNCATE_LENGTH, DROPOUT, BIDIRECTIONAL)
model = CNN(embeddings, padding_idx, 667, TRUNCATE_LENGTH, DROPOUT)
# Example of how to load a previously trained model
# model.load_state_dict(torch.load('lstm_saved_models/epoch1.pkl'))


# In[12]:


BATCH_SIZE = 20 # Normally use 20
NUM_EPOCHS = 50 # Normally use 50, but can stop early at 20
MARGINS = [0.2, 0.4, 0.6] # Some student on piazza said 0.2 worked really well
MARGIN = 0.2
LRS = [1e-3, 3e-4] # Taken from paper
LR = 1e-3

#SAVE_DIR = 'lstm_saved_models'
SAVE_DIR = 'cnn_saved_models'

train_model(train_dataset, dev_dataset, test_dataset, model, SAVE_DIR,
            num_epochs=NUM_EPOCHS, 
            margin=MARGIN, batch_size=BATCH_SIZE, lr=LR)

