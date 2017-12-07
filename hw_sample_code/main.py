from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils
import torch.utils.data
import gzip
from tqdm import tqdm

torch.manual_seed(1)

def get_embeddings_by_word(filename):
    word_embeddings = {}
    with gzip.open(filename, "r") as fh:
        for line in fh.readlines():
            split_line = line.strip().split(" ")
            word = split_line[0]
            vector = [float(num) for num in split_line[1:]]
            word_embeddings[word] = np.array(vector)
    return word_embeddings


def read_input(filename):
    with open(filename, "r") as fh:
        lines = fh.readlines()
    tokenizer = CountVectorizer().build_tokenizer()
    review_tokens_l = []
    scores = []
    for line in lines:
        score, review = line.strip().split(" ", 1)
        review_tokens_l.append(tokenizer(review))
        scores.append(int(score))
    return review_tokens_l, scores


def get_features(review_tokens_l, scores, embeddings_by_word):
    feature_vecs_l = list()
    clean_scores = list()
    for i in range(len(review_tokens_l)):
        review_tokens = review_tokens_l[i]
        score = scores[i]

        # List of n-dimensional embeddings
        embeddings_l = list()
        for token in review_tokens:
            if token in embeddings_by_word:
                embeddings_l.append(embeddings_by_word[token])

        if len(embeddings_l) == 0:
            continue

        feature_vec = np.array(embeddings_l).mean(axis=0)

        clean_scores.append(score)
        feature_vecs_l.append(feature_vec)

    return np.array(feature_vecs_l), np.array(clean_scores)


class ReviewSentimentsDataset(torch.utils.data.Dataset):
    def _get_dataset(self, features_mat, scores):
        dataset = []
        for i in range(len(scores)):
            dataset.append(dict(x=torch.FloatTensor(features_mat[i, :]), y=scores[i]))
        return dataset

    def __init__(self, filename, embeddings_by_word):
        review_tokens_l, scores = read_input(filename)
        features_mat, clean_scores = get_features(review_tokens_l, scores, embeddings_by_word)
        print("features_mat.shape", features_mat.shape, "clean_scores.shape", clean_scores.shape)
        self.dataset = self._get_dataset(features_mat, clean_scores)

    def get_dataset(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def run_epoch(data, is_training, model, optimizer):
    data_loader = torch.utils.data.DataLoader(data, batch_size=173,
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
        loss = torch.nn.NLLLoss()(out, y.long())

        _, predicted = torch.max(out.data, 1)
        total = total + y.size(0)
        right = right + (predicted == batch["y"]).sum()

        if is_training:
            loss.backward()
            optimizer.step()
        losses.append(loss.cpu().data[0])
    avg_loss = np.mean(losses)
    avg_accuracy = right / total
    return avg_loss, avg_accuracy


def train_model(train_data, dev_data, test_data, model, num_epochs=50, lr=1.0, weight_decay=0):
    print("start train_model")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(1, num_epochs + 1):
        print("epoch", epoch)
        train_loss, train_acc = run_epoch(train_data, True, model, optimizer)
        dev_loss, dev_acc = run_epoch(dev_data, False, model, optimizer)
        test_loss, test_acc = run_epoch(test_data, False, model, optimizer)
        print("train_loss", train_loss, "dev_loss", dev_loss, "test_loss", test_loss)
        print("train_acc", train_acc, "dev_acc", dev_acc, "test_acc", test_acc)

train_filename = "../data/stsa.binary.train"
dev_filename = "../data/stsa.binary.dev"
test_filename = "../data/stsa.binary.test"

print("getting embeddings_by_word")
embeddings_by_word = get_embeddings_by_word("../word_vectors.txt.gz")

print("instantiating datasets")
train_dataset = ReviewSentimentsDataset(train_filename, embeddings_by_word)
dev_dataset = ReviewSentimentsDataset(dev_filename, embeddings_by_word)
test_dataset = ReviewSentimentsDataset(test_filename, embeddings_by_word)

num_features = 300
# hidden_layer_sizes = [num_features/4, num_features/2, num_features, 2*num_features]
# lrs = [1e-5, 1e-3, 1e-1, 1e1]
# weight_decays = [1e-5, 1e-3, 1e1]

hidden_layer_sizes = [num_features/2]
lrs = [1e-3]
weight_decays = [1e-3]

for hidden_layer_size in hidden_layer_sizes:
    for lr in lrs:
        for weight_decay in weight_decays:
            print("PARAMS", "hidden_layer_size", hidden_layer_size,
                  "lr", lr, "weight_decay", weight_decay)
            print("instantiating the nn")
            model = torch.nn.Sequential(
                torch.nn.Linear(num_features, hidden_layer_size),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_layer_size, 2),
                torch.nn.LogSoftmax()
            )
            train_model(train_dataset, dev_dataset, test_dataset, model, lr=lr, weight_decay=weight_decay)
