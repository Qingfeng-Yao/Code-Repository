from datasets import Reuters_Dataset, Newsgroups20_Dataset, IMDB_Dataset

print('*** processing reuters ***')
nc_reuters = 6 # {0: earn, 1: acq, 2: crude, 3: trade, 4: money-fx, 5: interest, 6: ship}
dataset_reuters = Reuters_Dataset(root='../data', normal_class=nc_reuters, tokenizer='spacy',
                                use_tfidf_weights=False, append_sos=False, append_eos=False,
                                clean_txt=True)
print('train size: {}'.format(len(dataset_reuters.train_set)))
print('test size: {}'.format(len(dataset_reuters.test_set)))
print('test normal size: {}'.format(len(dataset_reuters.test_n_set)))
print('test anomalous size: {}'.format(len(dataset_reuters.test_a_set)))
print('vocab size: {}'.format(dataset_reuters.encoder.vocab_size))

# print('*** processing newsgroups20 ***')
# nc_newsgroups20 = 5 # {0: comp, 1: rec, 2: sci, 3: misc, 4: pol, 5: rel}
# dataset_newsgroups20 = Newsgroups20_Dataset(root='../data', normal_class=nc_newsgroups20, tokenizer='spacy',
#                                 use_tfidf_weights=False, append_sos=False, append_eos=False,
#                                 clean_txt=True)
# print('train size: {}'.format(len(dataset_newsgroups20.train_set)))
# print('test size: {}'.format(len(dataset_newsgroups20.test_set)))
# print('test normal size: {}'.format(len(dataset_newsgroups20.test_n_set)))
# print('test anomalous size: {}'.format(len(dataset_newsgroups20.test_a_set)))
# print('vocab size: {}'.format(dataset_newsgroups20.encoder.vocab_size))

# print('*** processing imdb ***')
# nc_imdb = 1 # {0: pos, 1: neg}
# dataset_imdb = IMDB_Dataset(root='../data', normal_class=nc_imdb, tokenizer='spacy',
#                                 use_tfidf_weights=False, append_sos=False, append_eos=False,
#                                 clean_txt=True)
# print('train size: {}'.format(len(dataset_imdb.train_set)))
# print('test size: {}'.format(len(dataset_imdb.test_set)))
# print('test normal size: {}'.format(len(dataset_imdb.test_n_set)))
# print('test anomalous size: {}'.format(len(dataset_imdb.test_a_set)))
# print('vocab size: {}'.format(dataset_imdb.encoder.vocab_size))