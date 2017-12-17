
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.utils.data
from torch import Tensor
import random


# In[ ]:


def read_text_tokenized(text_tokenized_file, truncate_length=100):
    # returns a dictionary of {question_id : (title, body)} key-value pairs
    print('read corpus', text_tokenized_file)
    question_id_to_title_body_tuple = {}
    for line in open(text_tokenized_file, 'r'):
        question_id, title, body = line.split('\t')
        title = title.split()[:truncate_length]
        body = body.split()[:truncate_length]
        if len(title) == 0:
            title = ['title']
        if len(body) == 0:
            body = ['body']
        question_id_to_title_body_tuple[question_id] = (title, body)
    return question_id_to_title_body_tuple


# In[ ]:


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


# In[ ]:


def read_ubuntu_eval_ids(eval_file, test_subset):
    # returns list of (question_id, candidate_ids, labels, bm25scores) tuples
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
        #assert(sum(labels)==len(positive_ids_set))
        eval_id_instances.append((qid, candidate_ids, labels, bm25scores))
        i += 1
        if (test_subset is not None) and i > test_subset:
            break
    return eval_id_instances


# In[ ]:


def read_android_eval_ids(pos_file_name, neg_file_name, test_subset=None):
    # returns list of (question_id, candidate_ids, labels) tuples
    # where all ids are strings, and labels is a list of binary positive/negative for each candidate in candidate_ids 
    eval_id_instances = []
    pos_file = open(pos_file_name)
    neg_file = open(neg_file_name)
    i = 0
    while True:
        # reads in one sample
        pos_file_line = pos_file.readline()
        if len(pos_file_line) == 0:
            break
        query, pos = pos_file_line.split()
        queries = [query]
        candidates = [pos]
        labels = [1]
        for j in range(100):
            query, neg = neg_file.readline().split()
            candidates.append(neg)
            queries.append(query)
            labels.append(0)
#         assert(len(set(queries)) == 1)
#         assert(len(candidates)==len(labels))
#         assert(labels[0]==1)
#         assert(sum(labels[1:])==0)
#         assert(candidates[0]==pos)
        eval_id_instances.append((query, candidates, labels))
        i +=1
        if (test_subset is not None) and i > test_subset:
            break
    return eval_id_instances


# In[ ]:


def read_word_embeddings(word_embeddings_file):
    #word_to_vec = {}
    print('reading word embedding file', word_embeddings_file)
    word_to_idx = {}
    embeddings = []
    for i, line in enumerate(open(word_embeddings_file)):
        split_line = line.split()
        word, vector = split_line[0], split_line[1:]
        vector = np.array([float(x) for x in vector])
        #word_to_vec[word] = vector
        word_to_idx[word] = i
        embeddings.append(vector)
    # last vector might not be full length
    if len(vector) < len(embeddings[0]):
        embeddings.pop()
    # add one for padding
    embeddings.append(np.zeros(len(embeddings[0])))
    padding_idx = i+1
    embeddings = np.array(embeddings)
    return word_to_idx, embeddings, padding_idx 


# In[ ]:


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, text_tokenized, word_to_idx, padding_idx, truncate):
        # text_tokenized: Either a string representing the filename of text_tokenized, OR
        #                 a precomputed id_to_question dictionary
        self.truncate = truncate
        if type(text_tokenized)==str:
            self.id_to_question = read_text_tokenized(text_tokenized, truncate_length=self.truncate)
        elif type(text_tokenized)==dict:
            self.id_to_question = text_tokenized
        self.word_to_idx = word_to_idx
        self.padding_idx = padding_idx
        #self.num_features = len(word_to_vec['.'])
    
    def get_sentence_matrix_embedding(self, words, num_words=100):
        # returns [num_words] np matrix
        # matrix may be padded
        sentence_mat = np.ones(num_words) * self.padding_idx
        i = 0
        mask = np.zeros(num_words)
        for word in words:
            # TODO: IS JUST SKIPPING THE WORD THE RIGHT APPROACH?
            if word in self.word_to_idx:
                sentence_mat[i] = self.word_to_idx[word]
            mask[i] = 1
            i += 1
            if i == num_words:
                break
        assert(sum(mask)>0), word
        return sentence_mat, mask

    def get_question_embedding(self, title_body_tuple):
        title_embedding, title_mask = self.get_sentence_matrix_embedding(title_body_tuple[0], self.truncate)
        body_embedding, body_mask = self.get_sentence_matrix_embedding(title_body_tuple[1], self.truncate)
        return Tensor(title_embedding), Tensor(body_embedding), Tensor(title_mask), Tensor(body_mask)
    
    def get_question_embeddings(self, title_body_tuples):
        num_questions = len(title_body_tuples)
        title_embeddings = np.zeros((num_questions, self.truncate))
        body_embeddings = np.zeros((num_questions, self.truncate))
        title_masks = np.zeros((num_questions, self.truncate))
        body_masks = np.zeros((num_questions, self.truncate))
        for i, (title, body) in enumerate(title_body_tuples):
            title_embedding, title_mask = self.get_sentence_matrix_embedding(title, self.truncate)
            body_embedding, body_mask = self.get_sentence_matrix_embedding(body, self.truncate)
            title_embeddings[i] = title_embedding
            body_embeddings[i] = body_embedding
            title_masks[i] = title_mask
            body_masks[i] = body_mask
        return Tensor(title_embeddings), Tensor(body_embeddings), Tensor(title_masks), Tensor(body_masks)
    
    def get_q_candidate_dict(self, q_id, candidate_ids):
        q = self.id_to_question[q_id]
        candidates = [self.id_to_question[id_] for id_ in candidate_ids]
        q_title_embedding, q_body_embedding, q_title_mask, q_body_mask = self.get_question_embedding(q)
        (candidate_title_embeddings, candidate_body_embeddings, 
         candidate_title_masks, candidate_body_masks) = self.get_question_embeddings(candidates)
        # candidate_*_embeddings is tensor of [num_cands x truncate_length]
        # q_*_embedding is tensor of [truncate_length]
        return dict(q_body=q_body_embedding.long(), q_title=q_title_embedding.long(),
            candidate_bodies=candidate_body_embeddings.long(), candidate_titles=candidate_title_embeddings.long(), 
            q_body_mask = q_body_mask, q_title_mask=q_title_mask, 
            candidate_body_masks=candidate_body_masks, candidate_title_masks=candidate_title_masks)


# In[ ]:


class TrainQuestionDataset(QuestionDataset):
    def __init__(self, text_tokenized_file, train_file, word_to_idx, padding_idx, truncate=100, test_subset=None):
        # test_subset: An integer representing the max number of training entries to consider.
        #              Used for quick debugging on a smaller subset of all training data.
        self.train_id_instances = read_train_ids(train_file, test_subset)
        QuestionDataset.__init__(self, text_tokenized_file, word_to_idx, padding_idx, truncate)
        
    def __len__(self):
        return len(self.train_id_instances)
        
    def __getitem__(self, index):
        (q_id, positive_id, negative_ids) = self.train_id_instances[index]
        negative_ids_sample = random.sample(negative_ids, 20)  # sample 20 random negatives
        candidate_ids = [positive_id]+negative_ids_sample
        return self.get_q_candidate_dict(q_id, candidate_ids)


# In[ ]:


class EvalQuestionDataset(QuestionDataset):
    def __init__(self, text_tokenized, eval_file, word_to_idx, padding_idx, truncate=100, test_subset=None):
        # test_subset: An integer representing the max number of entries to consider.
        #              Used for quick debugging on a smaller subset of all eval data.
        # text_tokenized: Either a string representing the filename of text_tokenized, OR
        #                 a precomputed id_to_question dictionary
        self.eval_id_instances = read_ubuntu_eval_ids(eval_file, test_subset)
        QuestionDataset.__init__(self, text_tokenized, word_to_idx, padding_idx, truncate)
    
    def __len__(self):
        return len(self.eval_id_instances)
    
    def __getitem__(self, index):
        (q_id, candidate_ids, labels, bm25scores) = self.eval_id_instances[index]
        item = self.get_q_candidate_dict(q_id, candidate_ids)
        item['labels'] = Tensor(labels)
        item['bm25scores'] = Tensor(bm25scores)
        return item    


# In[ ]:


class AndroidEvalQuestionDataset(QuestionDataset):
    # Same idea as UbuntuEval, only difference is text_tokenized corpus file will be different, and
    # no bm25scores. Not that we're using bm25 for UbuntuEval anyways.
    def __init__(self, text_tokenized, eval_pos_file, eval_neg_file, word_to_idx, padding_idx, truncate=100, test_subset=None):
        self.eval_id_instances = read_android_eval_ids(eval_pos_file, eval_neg_file, test_subset)
        QuestionDataset.__init__(self, text_tokenized, word_to_idx, padding_idx, truncate)
    
    def __len__(self):
        return len(self.eval_id_instances)
    
    def __getitem__(self, index):
        (q_id, candidate_ids, labels) = self.eval_id_instances[index]
        item = self.get_q_candidate_dict(q_id, candidate_ids)
        item['labels'] = Tensor(labels)
        return item    


# In[ ]:


class AndroidQuestionCorpusDataset(QuestionDataset):
    # Dataset where each data item is (question_title, question_body)
    def __init__(self, text_tokenized, word_to_idx, padding_idx, truncate=100, test_subset=22840):
        QuestionDataset.__init__(self, text_tokenized, word_to_idx, padding_idx, truncate)
        self.corpus = []
        i = 0
        for id_ in self.id_to_question:
            self.corpus.append(self.id_to_question[id_])
            i += 1
            if i > test_subset:
                break
        
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, index):
        q = self.corpus[index]
        q_title_embedding, q_body_embedding, q_title_mask, q_body_mask = self.get_question_embedding(q)
        return dict(q_body=q_body_embedding.long(), q_title=q_title_embedding.long(),
            q_body_mask = q_body_mask, q_title_mask=q_title_mask)


# In[ ]:


class TransferTrainQuestionDataset(torch.utils.data.Dataset):
    # Contains all the ubuntu dataset objects, plus a label of 'isUbuntu' sort of thing
    # Also all the android dataset objects
    def __init__(self, android_corpus, ubuntu_corpus, ubuntu_train_file, word_to_idx, padding_idx, truncate=100, test_subset=None):
        # Construct a mixed dataset
        # Where dataset[i = 0 to n_ubuntu-1] is ubuntu[i], and
        #       dataset[i = n_ubuntu to n_ubuntu+n_android-1] is android[i-n_ubuntu]
        self.truncate = truncate
        self.ubuntu_dataset = TrainQuestionDataset(ubuntu_corpus, ubuntu_train_file, word_to_idx, padding_idx, truncate=truncate, test_subset=test_subset)
        self.android_dataset = AndroidQuestionCorpusDataset(android_corpus, word_to_idx, padding_idx, truncate=truncate, test_subset=test_subset)
        self.n_ubuntu = len(self.ubuntu_dataset)
        self.n_android = len(self.android_dataset)
        
    def __len__(self):
        return self.n_ubuntu + self.n_android
    
    def __getitem__(self, index):
        if index < self.n_ubuntu:
            # return ubuntu instance
            item = self.ubuntu_dataset[index]
            item['isUbuntu'] = 1
        else:
            item = self.android_dataset[index - self.n_ubuntu]
            item['isUbuntu'] = 0
            # create dummy items to get batches in data_loader to play nice
            item['candidate_bodies'] = torch.zeros(21, self.truncate).long()
            item['candidate_titles'] = torch.zeros(21, self.truncate).long()
            item['candidate_body_masks'] = torch.zeros(21, self.truncate)
            item['candidate_title_masks'] = torch.zeros(21, self.truncate)
        return item


# In[ ]:


# WORD_EMBEDDINGS_FILE = 'askubuntu/vector/vectors_pruned.200.txt'
# UBUNTU_TEXT_TOKENIZED_FILE = 'askubuntu/text_tokenized.txt'
# UBUNTU_TRAIN_FILE = 'askubuntu/train_random.txt'
# UBUNTU_DEV_FILE = 'askubuntu/dev.txt'
# UBUNTU_TEST_FILE = 'askubuntu/test.txt'

# ANDROID_DEV_NEG_FILE = 'Android/dev.neg.txt'
# ANDROID_DEV_POS_FILE = 'Android/dev.pos.txt'
# ANDROID_TEST_NEG_FILE = 'Android/test.neg.txt'
# ANDROID_TEST_POS_FILE = 'Android/test.pos.txt'
# ANDROID_TEXT_TOKENIZED_FILE = 'Android/corpus.tsv'

# TRUNCATE_LENGTH = 100
# word_to_idx, embeddings, padding_idx = read_word_embeddings(WORD_EMBEDDINGS_FILE)


# In[ ]:


# # For doing ubuntu training alone (I.E. Part 1)
# ubuntu_train_dataset = TrainQuestionDataset(UBUNTU_TEXT_TOKENIZED_FILE, UBUNTU_TRAIN_FILE, word_to_idx, padding_idx, truncate=TRUNCATE_LENGTH)
# ubuntu_dev_dataset = EvalQuestionDataset(ubuntu_train_dataset.id_to_question, UBUNTU_DEV_FILE, word_to_idx, padding_idx, truncate=TRUNCATE_LENGTH, test_subset=1000)
# ubuntu_test_dataset = EvalQuestionDataset(ubuntu_train_dataset.id_to_question, UBUNTU_TEST_FILE, word_to_idx, padding_idx, truncate=TRUNCATE_LENGTH, test_subset=1000)


# In[ ]:


# # For doing android eval alone (I.E. Part 2a - baselines)
# android_dev_dataset = AndroidEvalQuestionDataset(ANDROID_TEXT_TOKENIZED_FILE, ANDROID_DEV_POS_FILE, ANDROID_DEV_NEG_FILE, 
#                                                  word_to_idx, padding_idx, truncate=100, test_subset=None)
# android_test_dataset = AndroidEvalQuestionDataset(android_dev_dataset.id_to_question, ANDROID_TEST_POS_FILE, ANDROID_TEST_NEG_FILE, 
#                                                  word_to_idx, padding_idx, truncate=100, test_subset=None)


# In[ ]:


# # For doing domain adaptation (I.E. Part2b)
# id_to_question = transfer_train_dataset.android_dataset.id_to_question
# transfer_train_dataset = TransferTrainQuestionDataset(id_to_question, UBUNTU_TEXT_TOKENIZED_FILE, 
#                                                       UBUNTU_TRAIN_FILE, word_to_idx, padding_idx, truncate=100, test_subset=None)
# android_dev_dataset = AndroidEvalQuestionDataset(transfer_train_dataset.android_dataset.id_to_question, ANDROID_DEV_POS_FILE, ANDROID_DEV_NEG_FILE, 
#                                                  word_to_idx, padding_idx, truncate=100, test_subset=None)
# android_test_dataset = AndroidEvalQuestionDataset(android_dev_dataset.id_to_question, ANDROID_DEV_POS_FILE, ANDROID_DEV_NEG_FILE, 
#                                                  word_to_idx, padding_idx, truncate=100, test_subset=None)

