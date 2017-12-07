import my_utils
from my_utils import Tag, IobTag, parse, Word, write_test_output_file
from collections import Counter
import numpy
import torch
import torch.utils.data
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils
from tqdm import tqdm
import nltk
from maxent_non_contextual_model import phi
from maxent_contextual_model import get_iob_tags_l

nltk.download("words")

class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, tagset_size, batch_size):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.tagset_size = tagset_size
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden(self.batch_size)

    def init_hidden(self, minibatch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, minibatch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, minibatch_size, self.hidden_dim)))

    def forward(self, sentence_inp):
        if len(sentence_inp.size())==3:
            num_sentences = self.batch_size
        else:
            num_sentences = 1
        self.hidden = self.init_hidden(num_sentences)
        sentence = sentence_inp.view((155, num_sentences, -1))
        lstm_out, self.hidden = self.lstm(sentence, self.hidden)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space)
        tag_scores_out = tag_scores.view((num_sentences, 155, self.tagset_size))
        return tag_scores_out

nltk.download("words")

class TagDataset(torch.utils.data.Dataset):
    def __init__(self, features_mat, target_vector):
        dataset = []
        for i in range(len(target_vector)):
            dataset.append(dict(x=torch.FloatTensor(features_mat[i]),
                                y=torch.FloatTensor(target_vector[i])))
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

def run_epoch(data, is_training, model, optimizer, class_weights, batch_size):
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                              shuffle=True, drop_last=True)
    losses = []
    if is_training:
        model.train()
    else:
        model.eval()

    total = 0.
    right = 0
    for batch in tqdm(data_loader):
        x = Variable(batch["x"])
        y = Variable(batch["y"])
        if is_training:
            optimizer.zero_grad()
        out = model(x)
        out = out.view((-1, 4))
        y = y.view((-1))
        #print(y.long())
        if class_weights is not None:
            loss = torch.nn.NLLLoss(weight=class_weights, ignore_index=4)(out, y.long())
        else:
            loss = torch.nn.NLLLoss()(out, y.long())

        _, predicted = torch.max(out.data, 1)
        total = total + y.size(0)
        right = right + (predicted == y.data.long()).sum()

        if is_training:
            loss.backward(retain_graph=True)
            optimizer.step()
        losses.append(loss.cpu().data[0])
    avg_loss = numpy.mean(losses)
    avg_accuracy = right / total
    return avg_loss, avg_accuracy


def train_model(train_data, model, batch_size=200, num_epochs=50, lr=1.0, weight_decay=0, class_weights=None):
    print("start train_model")
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(1, num_epochs + 1):
        print("epoch", epoch)
        train_loss, train_acc = run_epoch(train_data, True, model, optimizer, class_weights, batch_size)
        print("train_loss", train_loss)
        print("train_acc", train_acc)

def predict_proba(model, input_vec):
    proba = model(Variable(torch.FloatTensor(input_vec))).data

def predict_iob_tags(test_features_mat, model):
    prediction_vec = list()
    for sentence_vec in test_features_mat:
        output = model(Variable(torch.FloatTensor(sentence_vec)))
        predicted = torch.max(output.data, 2)[1][0]
        predicted = predicted.numpy()
        predicted[predicted == 3] = 2
        predicted_tags = [IobTag.get_tag(enc) for enc in predicted]
        prediction_vec.append(predicted_tags)
    return prediction_vec

def get_token_strs(test_features_mat, test_words_l, model):
    iob_tags_l = predict_iob_tags(test_features_mat, model)
    prediction_vec = list()
    current_gene_tag = Tag.gene1
    for i, test_words in enumerate(test_words_l):
        sequence = list()
        for j, test_word in enumerate(test_words):
            iob_tag = iob_tags_l[i][j]
            if iob_tag == IobTag.o:
                tag = Tag.tag
            elif iob_tag == IobTag.b:
                current_gene_tag = Tag.get_other_gene_tag(current_gene_tag)
                tag = current_gene_tag
            else:
                tag = current_gene_tag
            sequence.append("_".join([test_word, tag]))
        prediction_vec.append(" ".join(sequence))
    return prediction_vec

def get_class_weights(train_target_vec):
    train_target_vec_f = train_target_vec.flatten()
    class_cnts = Counter(train_target_vec_f)  #list of encoded labels
    class_weights = numpy.array(class_cnts.values(), dtype=numpy.float32)
    class_weights = sum(class_weights) / class_weights # [10, 10, 1] = [GENE1, GENE2, TAG]
    class_weights_tensor = torch.from_numpy(class_weights)
    return class_weights_tensor

def get_features_mat(words_l):
    num_features = len(phi("testword"))
    features_mat = []
    for words in words_l:
        sentence_mat = numpy.zeros((155, num_features))
        i = 0
        for word in words:
            sentence_mat[i] = phi(word)
            i += 1
            if i == 155:
                break
        features_mat.append(sentence_mat)
    return numpy.array(features_mat)

def get_target_vec(iob_tags_l):
    target_vec = list()
    for iob_tags in iob_tags_l:
        sentence_tags = numpy.ones(155)*(4)
        i = 0
        for iob_tag in iob_tags:
            sentence_tags[i] = IobTag.get_enc(iob_tag)
            i +=1
            if i == 155:
                break
        target_vec.append(sentence_tags)
    return numpy.array(target_vec)

def run(train_filename, test_filename, output_filename):
    train_ids, train_words_l, train_tags_l = my_utils.parse(train_filename)
    test_ids, test_words_l, ignore = my_utils.parse(test_filename)

    train_iob_tags_l = get_iob_tags_l(train_tags_l)
    train_features_mat = get_features_mat(train_words_l)
    train_target_vec = get_target_vec(train_iob_tags_l)
    train_dataset = TagDataset(train_features_mat, train_target_vec)

    n_features = len(phi('testword'))
    batch_size = 1000
    model = LSTMTagger(n_features, 10, 4, batch_size)
    lr = 1e-1
    weight_decay = 1e-3
    class_weights = get_class_weights(train_target_vec)
    train_model(train_dataset, model, num_epochs=7, lr=lr,
                weight_decay=weight_decay, class_weights=class_weights, batch_size=batch_size)

    test_features_mat = get_features_mat(test_words_l)
    test_token_strs = get_token_strs(test_features_mat, test_words_l, model)
    my_utils.write_test_output_file(output_filename, test_ids, test_token_strs)
