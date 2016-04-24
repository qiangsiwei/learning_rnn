# -*- coding: utf-8 -*-  

import os
import sys
import csv
import nltk
import time
import operator
import itertools
import numpy as np
from utils import *
from datetime import datetime
from rnn_theano import RNNTheano

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # 保存模型参数
            save_model_parameters_theano("../data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        for i in range(len(y_train)):
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


if __name__ == '__main__':
    
    _VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
    _HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
    _LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
    _NEPOCH = int(os.environ.get('NEPOCH', '100'))
    _MODEL_FILE = os.environ.get('MODEL_FILE')

    vocabulary_size = _VOCABULARY_SIZE
    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"

    with open('../data/reddit-comments-2015-08.csv', 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))
        
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())

    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
    t1 = time.time()
    model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
    t2 = time.time()
    print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

    if _MODEL_FILE != None:
        load_model_parameters_theano(_MODEL_FILE, model)

    train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)
