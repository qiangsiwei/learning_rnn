# -*- coding: utf-8 -*-  

import sys
import operator
import numpy as np
from utils import *
from datetime import datetime

class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # 初始化参数
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        T = len(x)
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((T, self.word_dim))
        for t in np.arange(T):
            # 使用x[t]，形同one-hot编码
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def bptt(self, x, y):
        T = len(y)
        # 前向传播算法
        o, s = self.forward_propagation(x)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # 反向传播算法，为避免梯度消失或爆炸只计算self.bptt_truncate步
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                # 更新模型参数
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        bptt_gradients = model.bptt(x, y)
        model_parameters = ['U', 'V', 'W']
        for pidx, pname in enumerate(model_parameters):
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                original_value = parameter[ix]
                # 计算梯度 (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = model.calculate_total_loss([x], [y])
                parameter[ix] = original_value - h
                gradminus = model.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                parameter[ix] = original_value
                # 反向传播梯度
                backprop_gradient = bptt_gradients[pidx][ix]
                # 计算相对误差 (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient) / (np.abs(backprop_gradient) + np.abs(estimated_gradient))
                if relative_error > error_threshold:
                    print
                    "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print
                    "+h Loss: %f" % gradplus
                    print
                    "-h Loss: %f" % gradminus
                    print
                    "Estimated_gradient: %f" % estimated_gradient
                    print
                    "Backpropagation gradient: %f" % backprop_gradient
                    print
                    "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print
            "Gradient check for parameter %s passed." % (pname)

    def sgd_step(self, x, y, learning_rate):
        # 计算参数梯度
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # 根据学习速率更新参数
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

# Outer SGD Loop
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # 调整学习速率
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        for i in range(len(y_train)):
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


if __name__ == '__main__':
    # X_train represent a sentence, y_train is the origin sentence shifted to right by 1 word.

    # sentence[0]: [12,31,3234,42,53]
    # X_train[0]: [12,31,3234,42]
    # y_train[1]: [31,3234,42,53]

    index_to_word = np.load('../data/index_to_word.npy')
    word_to_index = np.load('../data/word_to_index.npy')
    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')
    vocabulary_size = len(index_to_word)

    print "vocabulary size: %d" % vocabulary_size
    model = RNNNumpy(vocabulary_size)

    # test cross entropy loss
    print "Expected Loss for random predictions: %f" % np.log(vocabulary_size)
    print "Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000])

    # gradient checking
    # To avoid expensive calculations, a smaller vocabulary size is used for checking.
    grad_check_vocab_size = 100
    np.random.seed(10)
    model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
    model.gradient_check([0,1,2,3], [1,2,3,4])

    #train
    np.random.seed(10)
    # Train on a small subset of the data to see what happens
    model = RNNNumpy(vocabulary_size)
    losses = train_with_sgd(model, X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)
