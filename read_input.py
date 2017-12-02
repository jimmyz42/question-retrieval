
# coding: utf-8

# In[26]:


import numpy as np
import torch
import torch.utils.data
from torch import Tensor


# In[10]:


def read_text_tokenized(text_tokenized_file, truncate_length=100):
    # returns a dictionary of {question_id : (title, body)} key-value pairs
    question_id_to_title_body_tuple = {}
    for line in open(text_tokenized_file, 'r'):
        question_id, title, body = line.split('\t')
        question_id_to_title_body_tuple[question_id] = (title.split()[:truncate_length], 
                                                        body.split()[:truncate_length])
    return question_id_to_title_body_tuple


# In[11]:


def read_train_ids(train_file):
    # returns list of (question_id, positive_id, [negative_id, ...]) tuples
    # where all ids are strings
    train_id_instances = []
    for line in open(train_file):
        qid, positive_ids, negative_ids = line.split('\t')
        negative_ids = negative_ids.split()
        for positive_id in positive_ids.split():
            train_id_instances.append((qid, positive_id, negative_ids))
    return train_id_instances


# In[12]:


def make_word_to_vec_dict(word_embeddings_file):
    word_to_vec = {}
    for line in open(word_embeddings_file):
        split_line = line.split()
        word, vector = split_line[0], split_line[1:]
        vector = np.array([float(x) for x in vector])
        word_to_vec[word] = vector
    return word_to_vec


# In[13]:


word_embeddings_file = 'askubuntu/vector/vectors_pruned.200.txt'
word_to_vec = make_word_to_vec_dict(word_embeddings_file)


# In[39]:


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


# In[40]:


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, text_tokenized_file, train_file, truncate=100):
        self.truncate = truncate
        self.id_to_question = read_text_tokenized(text_tokenized_file, truncate_length=self.truncate)
        self.train_id_instances = read_train_ids(train_file)
        
    def __len__(self):
        return len(train_id_instances)
    
    def __getitem__(self, index):
        (q_id, positive_id, negative_ids) = self.train_id_instances[index]
        q_title, q_body = self.id_to_question[q_id]
        positive_title, positive_body = self.id_to_question[positive_id]
        negative_title_body_tuples = [self.id_to_question[neg_id] for neg_id in negative_ids]
        negative_bodies = [tup[1] for tup in negative_title_body_tuples]
        q_body_matrix = Tensor(get_sentence_matrix_embedding(q_body, self.truncate))
        positive_body_matrix = Tensor(get_sentence_matrix_embedding(positive_body, self.truncate))
        negative_body_matrices = [(get_sentence_matrix_embedding(neg_body, self.truncate)) for 
                             neg_body in negative_bodies]
        negative_body_matrices = Tensor(np.array(negative_body_matrices))
        # negative_body_matrices is tensor of [100 x truncate_length x 200]
        # q_body_matrix and positive_body_matrix are tensors of [truncate_length x 200]
        return dict(q=q_body_matrix, p=positive_body_matrix, negatives=negative_body_matrices)

dataset = QuestionDataset('askubuntu/text_tokenized.txt', 'askubuntu/train_random.txt', truncate=150)


# In[42]:


dataset[2]['q']

