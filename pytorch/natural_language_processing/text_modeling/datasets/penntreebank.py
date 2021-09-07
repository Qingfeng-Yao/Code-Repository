import torch.utils.data as data
from torchnlp.datasets import penn_treebank_dataset

import numpy as np 
import json
import os

class PennTreeBankDataset(data.Dataset):

	VOCABULARY_FILE = "penn-treebank/vocabulary.json"
	VOCAB_DICT = None
	LENGTH_PROB_FILE = "penn-treebank/length_prior.npy"

	def __init__(self, max_seq_len, train=False, val=False, test=False, root="data/", **kwargs):
		self.max_seq_len = max_seq_len
		self.dist_between_sents = int(self.max_seq_len / 10)
		self.is_train = train

		dataset = penn_treebank_dataset(root + "penn-treebank", train=train, dev=val, test=test)

		self.vocabulary = PennTreeBankDataset.get_vocabulary(root=root)
		self.index_to_word = {val: key for key, val in self.vocabulary.items()}

		words = [[]]
		for word_index, word in enumerate(dataset):
			if word == "</s>":
				words.append([])
			else:
				if word in self.vocabulary:
					words[-1].append(self.vocabulary[word])
				else:
					words[-1] += [self.vocabulary[c] for c in word]
				if word != "</s>":
					words[-1].append(self.vocabulary[" "])
				
		self.data = [np.array(sent) for sent in words if (len(sent) != 0 and len(sent)<self.max_seq_len)]
		self.total_tokens = sum(len(sent) for sent in self.data)

		print("Length of dataset(num of sents): ", len(self)) # num of sents
		print("total tokens: ", self.total_tokens)


	def __len__(self):
		return len(self.data)


	def __getitem__(self, idx): # get real sent length and padded sent content
		length = self.data[idx].shape[0]
		if length < self.max_seq_len:
			padded_data = np.concatenate([self.data[idx], np.zeros(self.max_seq_len-length, dtype=np.int32)], axis=0)
		else:
			padded_data = self.data[idx][:self.max_seq_len]
			length = min(self.max_seq_len, length)
		return padded_data, length


	@staticmethod
	def get_vocabulary(root="data/", **kwargs):
		if PennTreeBankDataset.VOCAB_DICT is None:
			if root is None:
				vocab_file = PennTreeBankDataset.VOCABULARY_FILE
			else:
				vocab_file = os.path.join(root, PennTreeBankDataset.VOCABULARY_FILE)
			if not os.path.isfile(vocab_file):
				PennTreeBankDataset.create_vocabulary(root=root)
			with open(vocab_file, "r") as f:
				PennTreeBankDataset.VOCAB_DICT = json.load(f)
		return PennTreeBankDataset.VOCAB_DICT


	@staticmethod
	def get_torchtext_vocab():
		return None


	@staticmethod
	def create_vocabulary(root="data/"):
		if root is None:
			root = ""
		dataset = penn_treebank_dataset(root + "penn-treebank", train=True, dev=False, test=False)
		all_words = [w for w in dataset]
		vocabulary = list(set([c for w in all_words for c in w])) + [" ", "<unk>", "</s>"]
		vocabulary = sorted(vocabulary)
		vocabulary = {vocabulary[i]: i for i in range(len(vocabulary))}
		with open(root + PennTreeBankDataset.VOCABULARY_FILE, "w") as f:
			json.dump(vocabulary, f, indent=4) # key: token, value: index

	@staticmethod
	def get_length_prior(max_seq_len, root="data/"):
		file_path = os.path.join(root, PennTreeBankDataset.LENGTH_PROB_FILE)
		if not os.path.isfile(file_path):
			train_dataset = PennTreeBankDataset(root=root, max_seq_len=1000, train=True)
			val_dataset = PennTreeBankDataset(root=root, max_seq_len=1000, val=True)
			sent_lengths = [d.shape[0] for d in train_dataset.data] + [d.shape[0] for d in val_dataset.data]
			sent_lengths_freq = np.bincount(np.array(sent_lengths))
			np.save(file_path, sent_lengths_freq)

		length_prior_count = np.load(file_path)
		length_prior_count = length_prior_count[:max_seq_len+1] + 1
		log_length_prior = np.log(length_prior_count) - np.log(length_prior_count.sum())
		return log_length_prior
		

if __name__ == '__main__':
	np.random.seed(42)
	dataset = PennTreeBankDataset(max_seq_len=288, train=True, val=False, test=False, root="../data/")
	data_loader = iter(data.DataLoader(dataset, 4))

	for data_index in range(10):
		sents = data_loader.next() # a tuple, first element shape [batch_size, max_len](dtype int), second element shape [batch_size, ]
		print(sents)
		if data_index > 4:
			break


