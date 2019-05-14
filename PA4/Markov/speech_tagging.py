import numpy as np
import time
import random
from hmm import HMM


def accuracy(predict_tagging, true_tagging):
	if len(predict_tagging) != len(true_tagging):
		return 0, 0, 0
	cnt = 0
	for i in range(len(predict_tagging)):
		if predict_tagging[i] == true_tagging[i]:
			cnt += 1
	total_correct = cnt
	total_words = len(predict_tagging)
	if total_words == 0:
		return 0, 0, 0
	return total_correct, total_words, total_correct*1.0/total_words


class Dataset:

	def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
		tags = self.read_tags(tagfile)
		data = self.read_data(datafile)
		self.tags = tags
		lines = []
		for l in data:
			new_line = self.Line(l)
			if new_line.length > 0:
				lines.append(new_line)
		if seed is not None: random.seed(seed)
		random.shuffle(lines)
		train_size = int(train_test_split * len(data))
		self.train_data = lines[:train_size]
		self.test_data = lines[train_size:]
		return

	def read_data(self, filename):
		"""Read tagged sentence data"""
		with open(filename, 'r') as f:
			sentence_lines = f.read().split("\n\n")
		return sentence_lines

	def read_tags(self, filename):
		"""Read a list of word tag classes"""
		with open(filename, 'r') as f:
			tags = f.read().split("\n")
		return tags

	class Line:
		def __init__(self, line):
			words = line.split("\n")
			self.id = words[0]
			self.words = []
			self.tags = []

			for idx in range(1, len(words)):
				pair = words[idx].split("\t")
				self.words.append(pair[0])
				self.tags.append(pair[1])
			self.length = len(self.words)
			return

		def show(self):
			print(self.id)
			print(self.length)
			print(self.words)
			print(self.tags)
			return


# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	###################################################
	# print(train_data[0].show())
	# print(tags)

	#obs_dict and state_dict
	state_dict = {}
	obs_dict = {}

	i = 0
	for k in tags:
		state_dict[k] = i
		i+=1

	#init obs_dict
	i=0
	for line in train_data:
		for word in line.words:
			if word not in obs_dict:
				obs_dict[word] = i
				i+=1
	M = i


	# N = len(tags)
	# # pi:
	# pi = np.zeros([N])
	# for sequence in train_data:
	# 	head_word_index = state_dict[sequence.tags[0]]
	# 	pi[head_word_index] +=1 
	# pi = pi * 1/N

	# # a:
	# A = np.zeros([N,N])
	# start_with_s = np.zeros(N,1)
	# for sequence in train_data:
	# 	for i in range(len(sequence.tags)):
	# 		if(i < len(sequence.tags)-1){
	# 			s = sequence.tags[i]
	# 			s2 = sequence.tags[i+1]
	# 			s_index = state_dict[s]
	# 			s2_index = state_dict[s2]
	# 			A[s][s2] += 1
	# 			# here we could divide sum of each row
	# 			start_with_s[s_idnex][0] += 1
	# 		}


	# A = A / start_with_s

	# # b:
	# B = np.zeros([N,M])
	# state_outcome_total = np.zeros(N,1)
	# for sequence in train_data:
	# 	for i in range(len(sequence.words):
	# 		state = sequence.tags[i]
	# 		observation = sequence.words[i]
	# 		state_index = state_dict[state]
	# 		observation_index = obs_dict[observation]
	# 		B[state_index,observation_index] += 1
	# 		state_outcome_total[state_index][0] +=1
	
	# B = B / state_outcome_total


	# init:
	N = len(tags)
	pi = np.zeros([N])
	A = np.zeros([N,N])
	start_with_s = np.zeros([N,1])
	B = np.zeros([N,M])
	state_outcome_total = np.zeros([N,1])

	for sequence in train_data:
		# pi:
		head_word_index = state_dict[sequence.tags[0]]
		pi[head_word_index] +=1
		for i in range(len(sequence.words)):
			s = sequence.tags[i]
			s_index = state_dict[s]
			if(i < len(sequence.tags)-1):
				s2 = sequence.tags[i+1]
				s2_index = state_dict[s2]
				A[s_index,s2_index] += 1
				# here we could divide sum of each row
				start_with_s[s_index,0] += 1

			observation = sequence.words[i]
			observation_index = obs_dict[observation]
			B[s_index,observation_index] += 1
			state_outcome_total[s_index,0] +=1

	pi = pi * 1 / N
	A = A / start_with_s
	B = B / state_outcome_total
	
	pi = np.nan_to_num(pi)
	A = np.nan_to_num(A)
	B = np.nan_to_num(B)

	# print(A)
	# print(B)
	# print(start_with_s)
	# print(state_outcome_total)



	model = HMM(pi,A,B,obs_dict,state_dict)

	return model


# TODO:
def speech_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	###################################################
	N,M = model.B.shape
	new_model = model
	new_column = 1e-6 * np.ones([N,1])
	new_feature_number = 0
	new_b = model.B
	new_obs_dict = model.obs_dict

	for sentence in test_data:
		for word in sentence.words:
			if word not in model.obs_dict:
				# add new features and set number of new features
				# add new column to b
				# sample : np.append(???,new_column,axis=1)
				new_b = np.append(new_b,new_column,axis=1)

				# add new features to obs_dict
				new_obs_dict[word] = len(new_b[0,:]) - 1

				# augment new features number
				new_feature_number += 1
	
	if new_feature_number != 0:
		new_model = HMM(model.pi, model.A, new_b, new_obs_dict, model.state_dict)
	
	for sentence in test_data:
		tag_row = new_model.viterbi(sentence.words)
		tagging.append(tag_row)

	return tagging

