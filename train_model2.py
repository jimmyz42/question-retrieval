
# coding: utf-8

# In[ ]:


# Part 2 stuff


# In[ ]:


#get_ipython().run_cell_magic(u'bash', u'', u'\njupyter nbconvert --to script read_input.ipynb')


# In[ ]:


import read_input
from read_input import TransferTrainQuestionDataset, AndroidEvalQuestionDataset, read_word_embeddings
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
import sys
from meter_auc import AUCMeter

from models import LSTM, CNN, evaluate, DomainClassifier


# In[ ]:


def masked_select_rows(matrix, mask, mask_value=1):
    # matrix is 2d tensor [n x m] (or variable containing 2d tensor)
    # mask is 1d tensor
    # returns matrix [new_n x m], with all the rows selected where mask=mask_value
    return matrix[torch.nonzero(mask==mask_value).view(-1)]


# In[ ]:


def run_eval_epoch(dataset, model, batch_size, save_path):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, drop_last=True)
    meter = AUCMeter()

    model.eval()
    for batch in tqdm(data_loader):

        # For evaluation:
        #    - Batch is only android eval examples
        #    - Examples have q_body, q_title, q_body_mask, q_title_mask, candidate_bodies, 
        #      candidate_titles, candidate_body_masks, candidate_title_masks, 
        #             AND label
        
        q_body = Variable(batch["q_body"]) # batch_size x truncate_length
        q_title = Variable(batch["q_title"])
        q_body_mask = Variable(batch["q_body_mask"]) # batch_size x truncate_length
        q_title_mask = Variable(batch["q_title_mask"])
        cand_bodies = Variable(batch["candidate_bodies"]) # batch_size x num_cands x truncate_length
        cand_titles = Variable(batch["candidate_titles"])
        cand_body_masks = Variable(batch["candidate_body_masks"]) # batch_size x num_cands x truncate_length
        cand_title_masks = Variable(batch["candidate_title_masks"])
        num_cands = cand_titles.size()[1]

        q_body_enc, q_title_enc = model(q_body, q_body_mask), model(q_title, q_title_mask) # output is batch_size  x enc_length
        cand_body_encs = model(cand_bodies.view(batch_size*num_cands, TRUNCATE_LENGTH), # output is (batch_size x num_cands) x enc_length
                               cand_body_masks.view(batch_size*num_cands, TRUNCATE_LENGTH)) 
        cand_title_encs = model(cand_titles.view(batch_size*num_cands, TRUNCATE_LENGTH),
                                cand_title_masks.view(batch_size*num_cands, TRUNCATE_LENGTH))
        q_enc = q_title_enc + q_body_enc / 2.0
        candidate_encs = cand_title_encs + cand_body_encs / 2.0
        
        candidate_encs = candidate_encs.view(batch_size, num_cands, -1) # batch_size x num_cands x enc_length
        query_encs = q_enc.view(batch_size, 1, -1).expand_as(candidate_encs) # batch_size x (num_cands) x enc_length
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-08)(candidate_encs, query_encs) # batch_size x (num_cands)
            
        labels = batch["labels"]
        meter.add(cos.data.view(-1), labels.view(-1))
    
    return meter.value(), meter.value(0.05) #return auc and auc(.05)


# In[ ]:


def run_train_epoch(dataset, encoder_model, domain_model, encoder_optimizer, domain_optimizer, batch_size, margin, _lambda, save_path):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, drop_last=True)
    similarity_losses = []
    domain_pred_losses = []
    encoder_model.train()
    domain_model.train()
    
    for batch in tqdm(data_loader):
        # For training: 
        #    Batch is mixed android and ubuntu examples
        #    All examples have a binary isUbuntu label, and the usual q_body, q_title, q_body_mask, q_title_mask
        #    For Android examples, the candidate_bodies, candidate_titles, candidate_body_masks, candidate_title_masks
        #      are all just tensors of all 0s, and SHOULD NOT BE USED!!! 
        
        q_body = Variable(batch["q_body"]) # batch_size x truncate_length
        q_title = Variable(batch["q_title"])
        q_body_mask = Variable(batch["q_body_mask"]) # batch_size x truncate_length
        q_title_mask = Variable(batch["q_title_mask"])
        ubuntu_cand_bodies = Variable(masked_select_rows(batch["candidate_bodies"], batch["isUbuntu"])) # num_ubuntu x num_cands x truncate_length
        ubuntu_cand_titles = Variable(masked_select_rows(batch["candidate_titles"], batch["isUbuntu"]))
        ubuntu_cand_body_masks = Variable(masked_select_rows(batch["candidate_body_masks"], batch["isUbuntu"])) # num_ubuntu x num_cands x truncate_length
        ubuntu_cand_title_masks = Variable(masked_select_rows(batch["candidate_title_masks"], batch["isUbuntu"]))
        num_ubuntu, num_cands = ubuntu_cand_titles.size()[0], ubuntu_cand_titles.size()[1]

        encoder_optimizer.zero_grad()
        domain_optimizer.zero_grad()
        q_body_enc, q_title_enc = encoder_model(q_body, q_body_mask), encoder_model(q_title, q_title_mask) # output is batch_size  x enc_length
        ubuntu_cand_body_encs = encoder_model(ubuntu_cand_bodies.view(num_ubuntu*num_cands, TRUNCATE_LENGTH), # output is (num_ubuntu x num_cands) x enc_length
                               ubuntu_cand_body_masks.view(num_ubuntu*num_cands, TRUNCATE_LENGTH)) 
        ubuntu_cand_title_encs = encoder_model(ubuntu_cand_titles.view(num_ubuntu*num_cands, TRUNCATE_LENGTH),
                                ubuntu_cand_title_masks.view(num_ubuntu*num_cands, TRUNCATE_LENGTH))
        q_enc = q_title_enc + q_body_enc / 2.0
        ubuntu_cand_encs = ubuntu_cand_title_encs + ubuntu_cand_body_encs / 2.0
    
        q_domain_pred = domain_model(q_enc)
        # do we also pass in the candidate stuff into the domain classifier? All of those will be ubuntu
        
        ubuntu_candidate_encs = ubuntu_cand_encs.view(num_ubuntu, num_cands, -1) # num_ubuntu x num_cands x enc_length
        ubuntu_q_enc = masked_select_rows(q_enc, batch["isUbuntu"])
        ubuntu_query_encs = ubuntu_q_enc.view(num_ubuntu, 1, -1).expand_as(ubuntu_candidate_encs) # num_ubuntu x (num_cands) x enc_length
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-08)(ubuntu_candidate_encs, ubuntu_query_encs) # num_ubuntu x (num_cands)
    

    ######################### CALCULATE LOSSES AND UPDATE #############################
        # is this right?????
        target = Variable(torch.zeros(num_ubuntu).long())
        similarity_training_loss = torch.nn.MultiMarginLoss(margin=margin)(cos, target)
        domain_pred_loss = torch.nn.CrossEntropyLoss()(q_domain_pred, Variable(batch["isUbuntu"]))
        total_encoder_loss = similarity_training_loss - _lambda*domain_pred_loss

        #similarity_training_loss.backward(retain_graph=False)
        #domain_pred_loss.backward(retain_graph=False)
        
        # only call backwards once on the overall loss, both optimizers step individually
        total_encoder_loss.backward()
        encoder_optimizer.step()
        domain_optimizer.step()

        similarity_losses.append(similarity_training_loss.data[0])
        domain_pred_losses.append(domain_pred_loss.data[0])
        ###########################################################################
    if save_path is not None:
        torch.save(encoder_model.state_dict(), save_path)
    avg_similarity_loss = np.mean(similarity_losses)
    avg_domain_loss = np.mean(domain_pred_losses)
    return avg_similarity_loss, avg_domain_loss


# In[ ]:


def train_model(train_data, dev_data, test_data, encoder_model, domain_model, save_dir=None, batch_size=50, 
                margin=0.2, _lambda=1e-3, num_epochs=50, lr=1.0):
    if (save_dir is not None) and (not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    print("start train_model")
    print("****************************************")
    print("Batch size: {}, margin: {}, num_epochs: {}, lr: {}".format(batch_size, margin, num_epochs, lr))
    print("Encoder Model", encoder_model)
    print("Domain classifier model", domain_model)
    print("*****************************************")
    encoder_parameters = filter(lambda p: p.requires_grad, encoder_model.parameters())
    # encoder has positive learning rate, domain has negative (because domain loss is subtracted in total loss)
    encoder_optimizer = torch.optim.Adam(encoder_parameters, lr=lr)
    domain_optimizer = torch.optim.Adam(domain_model.parameters(), lr=-lr)
    
    result_table = PrettyTable(["Epoch", "train similarity loss", "train domain pred loss", 
                                "dev auc", "dev auc(0.05)", "test auc", "test auc(0.05)"])
    for epoch in range(1, num_epochs + 1):
        print("epoch", epoch)
        if save_dir is None:
            save_path = None
        else:
            save_path = os.path.join(save_dir, 'epoch{}.pkl'.format(epoch))

        similarity_loss, domain_loss = run_train_epoch(train_data, encoder_model, domain_model, encoder_optimizer, 
                                                       domain_optimizer, batch_size, margin, _lambda, save_path)
        dev_auc, dev_auc05 = run_eval_epoch(dev_data, encoder_model, batch_size, save_path)
        test_auc, test_auc05 = run_eval_epoch(test_data, encoder_model, batch_size, save_path)
        result_table.add_row(
                            [ epoch ] +
                            [ "%.3f" % x for x in [similarity_loss, domain_loss] + [ dev_auc, dev_auc05 ] +
                                        [ test_auc, test_auc05 ] ])
        print("{}".format(result_table))
        sys.stdout.flush()


# In[ ]:


WORD_EMBEDDINGS_FILE = 'vectors_stackexchange.txt'
UBUNTU_TEXT_TOKENIZED_FILE = 'askubuntu/text_tokenized.txt'
UBUNTU_TRAIN_FILE = 'askubuntu/train_random.txt'
UBUNTU_DEV_FILE = 'askubuntu/dev.txt'
UBUNTU_TEST_FILE = 'askubuntu/test.txt'

ANDROID_DEV_NEG_FILE = 'Android/dev.neg.txt'
ANDROID_DEV_POS_FILE = 'Android/dev.pos.txt'
ANDROID_TEST_NEG_FILE = 'Android/test.neg.txt'
ANDROID_TEST_POS_FILE = 'Android/test.pos.txt'
ANDROID_TEXT_TOKENIZED_FILE = 'Android/corpus.tsv'

TRUNCATE_LENGTH = 100
word_to_idx, embeddings, padding_idx = read_word_embeddings(WORD_EMBEDDINGS_FILE)


# In[ ]:


# For doing domain adaptation (I.E. Part2b)
transfer_train_dataset = TransferTrainQuestionDataset(ANDROID_TEXT_TOKENIZED_FILE, UBUNTU_TEXT_TOKENIZED_FILE, 
                                                      UBUNTU_TRAIN_FILE, word_to_idx, padding_idx, 
                                                      truncate=100, test_subset=2000)
android_dev_dataset = AndroidEvalQuestionDataset(transfer_train_dataset.android_dataset.id_to_question, 
                                                 ANDROID_DEV_POS_FILE, ANDROID_DEV_NEG_FILE, 
                                                 word_to_idx, padding_idx, truncate=100, test_subset=200)
android_test_dataset = AndroidEvalQuestionDataset(android_dev_dataset.id_to_question, ANDROID_DEV_POS_FILE, 
                                                  ANDROID_DEV_NEG_FILE, 
                                                 word_to_idx, padding_idx, truncate=100, test_subset=200)


# In[ ]:


BATCH_SIZE = 20 # Normally use 16
NUM_EPOCHS = 50 # Normally use 50, but can stop early at 20
MARGINS = [0.2, 0.4, 0.6] # Some student on piazza said 0.2 worked really well
MARGIN = 0.2
LRS = [1e-3, 3e-4] # Taken from paper
LR = 1e-3
LAMBDA = 1e-3
SAVE_DIR = 'domain_saved_models'


# In[ ]:


DROPOUT = 0.1
BIDIRECTIONAL = False
ENCODING_LENGTH = 240
encoder_model = LSTM(embeddings, padding_idx, ENCODING_LENGTH, 1, TRUNCATE_LENGTH, DROPOUT, BIDIRECTIONAL)

HIDDEN_DIM_1 = 300
HIDDEN_DIM_2 = 150
domain_model = DomainClassifier(ENCODING_LENGTH, HIDDEN_DIM_1, HIDDEN_DIM_2)


# In[ ]:


train_model(transfer_train_dataset, android_dev_dataset, android_test_dataset, 
            encoder_model, domain_model, 
            save_dir=SAVE_DIR, batch_size=BATCH_SIZE, margin=MARGIN, _lambda=LAMBDA, num_epochs=NUM_EPOCHS, lr=LR)

