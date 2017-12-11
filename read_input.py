
# coding: utf-8

# In[5]:


import numpy as np
import torch
import torch.utils.data
from torch import Tensor
import random


# In[6]:


def read_text_tokenized(text_tokenized_file, truncate_length=100):
    # returns a dictionary of {question_id : (title, body)} key-value pairs
    question_id_to_title_body_tuple = {}
    for line in open(text_tokenized_file, 'r'):
        question_id, title, body = line.split('\t')
        question_id_to_title_body_tuple[question_id] = (title.split()[:truncate_length], 
                                                        body.split()[:truncate_length])
    return question_id_to_title_body_tuple


# In[7]:


def read_train_ids(train_file, test_subset):
    # returns list of (question_id, positive_id, [negative_id, ...]) tuples
    # where all ids are strings
    train_id_instances = []
    i = 0
    for line in open(train_file):
        qid, positive_ids, negative_ids = line.split('\t')
        negative_ids = negative_ids.split()
        for positive_id in positive_ids.split():
            train_id_instances.append((qid, positive_id, negative_ids))
            i += 1
        if (test_subset is not None) and i > test_subset:
            break
    return train_id_instances


# In[75]:


def read_eval_ids(eval_file, test_subset):
    # returns list of (question_id, candidate_ids, labels) tuples
    # where all ids are strings, and labels is a list of binary positive/negative for each candidate in candidate_ids
    eval_id_instances = []
    i = 0
    for line in open(eval_file):
        qid, positive_ids, candidate_ids, bm25scores = line.split('\t')
        positive_ids_set = set(positive_ids.split())
        candidate_ids = candidate_ids.split()
        bm25scores = [float(score) for score in bm25scores.split()]
        if len(positive_ids_set) == 0:
            continue
        labels = [1 if cid in positive_ids_set else 0 for cid in candidate_ids]
        assert(sum(labels)==len(positive_ids_set))
        eval_id_instances.append((qid, candidate_ids, labels, bm25scores))
        i += 1
        if (test_subset is not None) and i > test_subset:
            break
    return eval_id_instances


# In[8]:


def make_word_to_vec_dict(word_embeddings_file):
    word_to_vec = {}
    for line in open(word_embeddings_file):
        split_line = line.split()
        word, vector = split_line[0], split_line[1:]
        vector = np.array([float(x) for x in vector])
        word_to_vec[word] = vector
    return word_to_vec


# In[10]:


def get_sentence_matrix_embedding(words, num_words=100):
    # returns [num_words x length_embedding] np matrix
    # matrix may be padded
    if len(words) >  num_words:
        # we shouldn't be printing here because we should have truncated already
        print(len(words))
    num_features = len(word_to_vec['.'])
    sentence_mat = np.zeros((num_words, num_features))
    i = 0
    for word in words:
        # TODO: IS JUST SKIPPING THE WORD THE RIGHT APPROACH?
        if word in word_to_vec:
            sentence_mat[i] = word_to_vec[word]
        i += 1
        if i == num_words:
            break
    return sentence_mat


# In[9]:


word_embeddings_file = 'askubuntu/vector/vectors_pruned.200.txt'
word_to_vec = make_word_to_vec_dict(word_embeddings_file)


# In[77]:


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, text_tokenized, truncate):
        # text_tokenized: Either a string representing the filename of text_tokenized, OR
        #                 a precomputed id_to_question dictionary
        self.truncate = truncate
        if type(text_tokenized)==str:
            self.id_to_question = read_text_tokenized(text_tokenized, truncate_length=self.truncate)
        elif type(text_tokenized)==dict:
            self.id_to_question = text_tokenized
        self.num_features = len(word_to_vec['.'])
    
    def get_question_embedding(self, title_body_tuple):
        title_embedding = Tensor(get_sentence_matrix_embedding(title_body_tuple[0], self.truncate))
        body_embedding = Tensor(get_sentence_matrix_embedding(title_body_tuple[1], self.truncate))
        return title_embedding, body_embedding
    
    def get_question_embeddings(self, title_body_tuples):
        num_questions = len(title_body_tuples)
        title_embeddings = np.zeros((num_questions, self.truncate, self.num_features))
        body_embeddings = np.zeros((num_questions, self.truncate, self.num_features))
        for i, (title, body) in enumerate(title_body_tuples):
            title_embeddings[i] = get_sentence_matrix_embedding(title, self.truncate)
            body_embeddings[i] = get_sentence_matrix_embedding(body, self.truncate)
        return Tensor(title_embeddings), Tensor(body_embeddings)


# In[78]:


class TrainQuestionDataset(QuestionDataset):
    def __init__(self, text_tokenized_file, train_file, truncate=100, test_subset=None):
        # test_subset: An integer representing the max number of training entries to consider.
        #              Used for quick debugging on a smaller subset of all training data.
        self.train_id_instances = read_train_ids(train_file, test_subset)
        QuestionDataset.__init__(self, text_tokenized_file, truncate)
        
    def __len__(self):
        return len(self.train_id_instances)
    
    def __getitem__(self, index):
        (q_id, positive_id, negative_ids) = self.train_id_instances[index]
        q = self.id_to_question[q_id]
        #p = self.id_to_question[positive_id]
        negative_ids_sample = random.sample(negative_ids, 20)  # sample 20 random negatives
        candidate_ids = [positive_id]+negative_ids_sample
        candidates = [self.id_to_question[id_] for id_ in candidate_ids]
        q_title_embedding, q_body_embedding = self.get_question_embedding(q)
        # candidate_body_embeddings is tensor of [num_cands x truncate_length x 200]
        # q_body_embedding and p_body_embedding are tensors of [truncate_length x 200]
        candidate_title_embeddings, candidate_body_embeddings = self.get_question_embeddings(candidates)
        return dict(q_body=q_body_embedding, q_title=q_title_embedding, 
                    candidate_bodies=candidate_body_embeddings, candidate_titles=candidate_title_embeddings)


# In[82]:


class EvalQuestionDataset(QuestionDataset):
    def __init__(self, text_tokenized, eval_file, truncate=100, test_subset=None):
        # test_subset: An integer representing the max number of entries to consider.
        #              Used for quick debugging on a smaller subset of all eval data.
        # text_tokenized: Either a string representing the filename of text_tokenized, OR
        #                 a precomputed id_to_question dictionary
        self.eval_id_instances = read_eval_ids(eval_file, test_subset)
        QuestionDataset.__init__(self, text_tokenized, truncate)
    
    def __len__(self):
        return len(self.eval_id_instances)
    
    def __getitem__(self, index):
        (q_id, candidate_ids, labels, bm25scores) = self.eval_id_instances[index]
        q = self.id_to_question[q_id]
        candidates = [self.id_to_question[id_] for id_ in candidate_ids]
        q_title_embedding, q_body_embedding = self.get_question_embedding(q)
        candidate_title_embeddings, candidate_body_embeddings = self.get_question_embeddings(candidates)
        # candidate_body_embedding is tensor of [num_cands x truncate_length x 200]
        # q_body_embedding is tensor of [truncate_length x 200]
        return dict(q_body=q_body_embedding, q_title=q_title_embedding, 
                    candidate_bodies=candidate_body_embeddings, candidate_titles=candidate_title_embeddings, 
                   labels=Tensor(labels), bm25scores=Tensor(bm25scores))


# In[80]:


# TEXT_TOKENIZED_FILE = 'askubuntu/text_tokenized.txt'
# TRAIN_FILE = 'askubuntu/train_random.txt'
# DEV_FILE = 'askubuntu/dev.txt'
# TEST_FILE = 'askubuntu/test.txt'

# TRUNCATE_LENGTH = 150
# train_dataset = TrainQuestionDataset(TEXT_TOKENIZED_FILE, TRAIN_FILE, truncate=TRUNCATE_LENGTH, test_subset=9)


# In[90]:


# dev_dataset = EvalQuestionDataset(train_dataset.id_to_question, DEV_FILE, truncate=TRUNCATE_LENGTH, test_subset=9)
# test_dataset = EvalQuestionDataset(train_dataset.id_to_question, TEST_FILE, truncate=TRUNCATE_LENGTH, test_subset=9)

