# -*- coding: utf-8 -*- 

import os
import re
import json
import numpy as np
from keras.preprocessing import sequence
from keras.models import Graph, slice_X, model_from_json
from keras.layers import recurrent, Dropout, TimeDistributedDense

class WordTable(object):
	def __init__(self):
		self.word2index = {}
		self.index2word = {}
		self.capacity = 0
	
	def parse(self, filename, sentence_num):
		for line in open(filename).readlines()[:sentence_num]:
			for word in line.strip().decode('utf-8'):
				if not word in self.word2index:
					self.capacity += 1
					self.word2index[word] = self.capacity
					self.index2word[self.capacity] = word
		self.capacity += 1

	def encode(self, sentence):
		return [self.word2index.get(word, 0) for word in sentence]

	def decode(self, sentence):
		return [self.index2word.get(index, '') for word in sentence]

	def to_json(self):
		return json.dumps({'capacity':self.capacity, 'word2index':self.word2index, 'index2word':self.index2word})

	def from_json(self, json_content):
		self.capacity = json_content['capacity']
		self.word2index = json_content['word2index']
		self.index2word = json_content['index2word']


def train_breaker(datafilename, sentence_num=1000, puncs=u',，.。!！?？', \
			RNN=recurrent.GRU, HIDDEN_SIZE=128, EPOCH_SIZE=10, validate=True):
	wordtable = WordTable()
	wordtable.parse(datafilename, sentence_num)

	X, Y = [], []
	for line in open(datafilename).readlines()[:sentence_num]:
		line = line.strip().decode('utf-8')
		line = re.sub(ur'(^[{0}]+)|([{0}]+$)'.format(puncs),'',line)
		words = wordtable.encode(re.sub(ur'[{0}]'.format(puncs),'',line))
		breaks = re.sub(ur'0[{0}]+'.format(puncs),'1',re.sub(ur'[^{0}]'.format(puncs),'0',line))
		if len(words) >= 30 and len(words) <= 50 and breaks.count('1') >= 4:
			x = np.zeros((len(words), wordtable.capacity), dtype=np.bool)
			y = np.zeros((len(breaks), 2), dtype=np.bool)
			for idx in xrange(len(words)):
				x[idx][words[idx]] = True
				y[idx][int(breaks[idx])] = True
			X.append(x)
			Y.append(y)
	print 'total sentence: ', len(X)

	if validate:
		# Set apart 10% for validation
		split_at = len(X) - len(X)/10
		X_train, X_val = X[:split_at], X[split_at:]
		y_train, y_val = Y[:split_at], Y[split_at:]
	else:
		X_train, y_train = X, Y

	model = Graph()
	model.add_input(name='input', input_shape=(None, wordtable.capacity))
	model.add_node(RNN(HIDDEN_SIZE, return_sequences=True), name='forward', input='input')
	model.add_node(TimeDistributedDense(2, activation='softmax'), name='softmax', input='forward')
	model.add_output(name='output', input='softmax')
	model.compile('adam', {'output': 'categorical_crossentropy'})

	for epoch in xrange(EPOCH_SIZE):
		print "epoch: ", epoch
		for idx, (seq, label) in enumerate(zip(X_train, y_train)):
			loss, accuracy = model.train_on_batch({'input':np.array([seq]), 'output':np.array([label])}, accuracy=True)
			if idx % 20 == 0:
				print "\tidx={0}, loss={1}, accuracy={2}".format(idx, loss, accuracy)

	if validate:
		_Y, _P = [], []
		for (seq, label) in zip(X_val, y_val):
			y = label.argmax(axis=-1)
			p = model.predict({'input':np.array([seq])})['output'][0].argmax(axis=-1)
			_Y.extend(list(y))
			_P.extend(list(p))
		_Y, _P = np.array(_Y), np.array(_P)
		print "should break right: ", ((_P == 1)*(_Y == 1)).sum()
		print "should break wrong: ", ((_P == 0)*(_Y == 1)).sum()
		print "should not break right: ", ((_P == 0)*(_Y == 0)).sum()
		print "should not break wrong: ", ((_P == 1)*(_Y == 0)).sum()

	with open('wordtable_json.txt','w') as wordtable_file:
		wordtable_file.write(wordtable.to_json())
	with open('model_json.txt','w') as model_file:
		model_file.write(model.to_json())
	model.save_weights('model_weights.h5', overwrite=True)


if __name__ == '__main__':
	puncs = u',，.。!！?？'
	SENTENCE_NUM, TEST_NUM = 10000, 1000
	datafilename = ""

	train_breaker(datafilename=datafilename, sentence_num=SENTENCE_NUM, RNN=recurrent.SimpleRNN, EPOCH_SIZE=10) # Replace with SimpleRNN, LSTM, GRU

	with open('model_json.txt', 'r') as model_file:
		model = model_from_json(model_file.read())
	model.load_weights('model_weights.h5')
	wordtable = WordTable()
	with open('wordtable_json.txt', 'r') as wordtable_file:
		wordtable.from_json(json.loads(wordtable_file.read()))

	for sentence in open(datafilename).readlines()[SENTENCE_NUM:SENTENCE_NUM+TEST_NUM]:
		sentence = re.sub(' ','',sentence.strip()).decode('utf-8')
		sentence = re.sub(ur'[{0}]'.format(puncs),'',sentence)
		words = wordtable.encode(sentence)
		X = np.zeros((len(words), wordtable.capacity), dtype=np.bool)
		for i, idx in enumerate(words):
			X[i][idx] = True
		breaks = model.predict({'input':np.array([X])})['output'][0].argmax(axis=-1)
		print ''.join([sentence[idx]+('|' if label else '') for idx, label in enumerate(breaks)])

