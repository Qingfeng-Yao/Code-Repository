from torchnlp.word_to_vector import GloVe

word_vectors_cache = './'

for embedding_size in (50, 100, 200, 300):
    GloVe(name='6B', dim=embedding_size, cache=word_vectors_cache)
