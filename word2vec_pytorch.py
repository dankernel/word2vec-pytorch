
import sys
import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F

def readFile(path):
    ret = []
    f = open(path, "r")
    for line in f:
        l = line.split()
        ret.append(l)

    return ret

tokenized_corpus = readFile('words.txt')

vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)

window_size = 2
idx_pairs = []

# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

embedding_dims = 3
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 100
learning_rate = 0.001

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:

        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        print('x')
        print(x)
        print('y_true')
        print(y_true)

        print('W1')
        print(W1)
        print('W2')
        print(W2)

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)

        print('z1')
        print(z1)

        print('z2')
        print(z2)

        
        log_softmax = F.log_softmax(z2, dim=0)
        loss = F.nll_loss(log_softmax.view(1,-1), y_true)

        print('log_softmax')
        print(log_softmax)

        exit()

        # loss_val += loss.data
        loss_val += loss.item()
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
        
        if epo % 100 == 0:    
            print('Loss at epo', epo, loss_val/len(idx_pairs))

print(vocabulary)
print(W2)

# readFile('words.txt')


