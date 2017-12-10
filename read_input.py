
import numpy as np
import torch
import torch.utils.data
from torch import Tensor

def read_text_tokenized(text_tokenized_file, truncate_length=100):
    # returns a dictionary of {question_id : (title, body)} key-value pairs
    question_id_to_title_body_tuple = {}
    for line in open(text_tokenized_file, 'r'):
        question_id, title, body = line.split('\t')
        question_id_to_title_body_tuple[question_id] = (title.split()[:truncate_length], 
                                                        body.split()[:truncate_length])
    return question_id_to_title_body_tuple

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

def make_word_to_vec_dict(word_embeddings_file):
    word_to_vec = {}
    for line in open(word_embeddings_file):
        split_line = line.split()
        word, vector = split_line[0], split_line[1:]
        vector = np.array([float(x) for x in vector])
        word_to_vec[word] = vector
    return word_to_vec

word_embeddings_file = 'askubuntu/vector/vectors_pruned.200.txt'
word_to_vec = make_word_to_vec_dict(word_embeddings_file)

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

class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, text_tokenized_file, train_file, truncate=100):
        # id_to_question is an optional pre-computed id_to_question
        self.truncate = truncate
        self.id_to_question = read_text_tokenized(text_tokenized_file, truncate_length=self.truncate)
        self.train_id_instances = read_train_ids(train_file)
        self.num_features = len(word_to_vec['.'])
        
    def __len__(self):
        return len(self.train_id_instances)
    
#     def get_question_embedding(self, title_body_tuple):
#         title_embedding = Tensor(get_sentence_matrix_embedding(title_body_tuple[0], self.truncate))
#         body_embedding = Tensor(get_sentence_matrix_embedding(title_body_tuple[1], self.truncate))
#         return title_embedding, body_embedding
    
    def get_question_embeddings(self, title_body_tuples):
        num_questions = len(title_body_tuples)
        title_embeddings = np.zeros((num_questions, self.truncate, self.num_features))
        body_embeddings = np.zeros((num_questions, self.truncate, self.num_features))
        for i, (title, body) in enumerate(title_body_tuples):
            title_embeddings[i] = get_sentence_matrix_embedding(title, self.truncate)
            body_embeddings[i] = get_sentence_matrix_embedding(body, self.truncate)
        return Tensor(title_embeddings), Tensor(body_embeddings)
    
    def __getitem__(self, index):
        (q_id, positive_id, negative_ids) = self.train_id_instances[index]
        q = self.id_to_question[q_id]
        p = self.id_to_question[positive_id]
        negatives = [self.id_to_question[neg_id] for neg_id in negative_ids]
        q_title_embedding, q_body_embedding = self.get_question_embeddings([q])
        p_title_embedding, p_body_embedding = self.get_question_embeddings([p])
        neg_title_embeddings, neg_body_embeddings = self.get_question_embeddings(negatives)
        # negative_body_matrices is tensor of [num_negs x truncate_length x 200]
        # q_body_matrix and positive_body_matrix are tensors of [1 x truncate_length x 200]
        return dict(q_body=q_body_embedding, q_title=q_title_embedding, 
                    p_body=p_body_embedding, p_title=p_title_embedding, 
                    neg_bodies=neg_body_embeddings, neg_titles=neg_title_embeddings)

#dataset = QuestionDataset('askubuntu/text_tokenized.txt', 'askubuntu/train_random.txt', truncate=150)
