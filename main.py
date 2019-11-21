
import numpy as np
import sys
import torch

from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F

WINDOW_SIZE = 5

def softmax(a) :
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def readFile(path):
    f = open(path, "r")
    ret = []
    for line in f:

        l = line.split()
        ret.append(l)

    return ret

def trainGenerator(sentences, uniqWords):

    trainPairs = []
    uniqWordSize = len(uniqWords)

    for sentence in sentences:
        for i in range(len(sentence)):

            index = uniqWords.index(sentence[i])

            oneHotVector = np.zeros(uniqWordSize)
            oneHotVector[index] = 1

            for j in range(-WINDOW_SIZE//2 + 1, WINDOW_SIZE//2 + 1):
                if 0 <= index + j and index + j < uniqWordSize and index + j != index:
                    trainPairs.append([oneHotVector, index + j])

    return trainPairs

def main():

    # Read sentences
    sentences = readFile("words2.txt")

    # Make uniq words list
    words = []
    uniqWords = []
    for sentence in sentences:
        for word in sentence:
            words.append(word)
            if word not in uniqWords:
                uniqWords.append(word)
    print(uniqWords)
    uniqWordSize = len(uniqWords)

    # Make trainPairs
    trainPairs = trainGenerator(sentences, uniqWords)

    dims = 5
    W1 = Variable(torch.randn(dims, uniqWordSize).float(), requires_grad=True)
    W2 = Variable(torch.randn(uniqWordSize, dims).float(), requires_grad=True)

    epo = 1001

    for i in range(epo):
        avg_loss = 0
        samples = 0
        for x, y in trainPairs:
            x = Variable(torch.from_numpy(x)).float()
            y = Variable(torch.from_numpy(np.array([y])).long())

            samples += len(y)

            a1 = torch.matmul(W1, x)
            a2 = torch.matmul(W2, a1)

            logSoftmax = F.log_softmax(a2, dim=0)
            loss = F.nll_loss(logSoftmax.view(1,-1), y)
            loss.backward()

            avg_loss += loss.item()

            W1.data -= 0.002 * W1.grad.data
            W2.data -= 0.002 * W2.grad.data

            W1.grad.data.zero_()
            W2.grad.data.zero_()

            if i != 0 and 100 < i and i % 100 == 0:
                print(avg_loss / samples)

    parisVecter = W1[:, uniqWords.index('paris')].data.numpy()
    context_to_predict = parisVecter
    hidden = Variable(torch.from_numpy(context_to_predict)).float()
    a = torch.matmul(W2, hidden)
    probs = F.softmax(a, dim=0).data.numpy()
    for context, prob in zip(uniqWords, probs):
        print(f'{context}: {prob:.2f}')

if __name__ == '__main__':
    main()

