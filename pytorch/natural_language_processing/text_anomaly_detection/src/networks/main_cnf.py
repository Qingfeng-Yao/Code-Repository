from .cnf_Net import CNFNet
from utils.word_vectors import load_word_vectors


def build_network(net_name, dataset, pretrained_model, embedding_size, word_vectors_cache='../data/word_vectors_cache', num_dimensions=None, encoding_params=None, coupling_hidden_size=1024, coupling_hidden_layers=2, coupling_num_flows=1, coupling_num_mixtures=64, coupling_dropout=0.0, coupling_input_dropout=0.0, max_seq_len=None, use_time_embed=False):
    """Builds the neural network."""
    if pretrained_model is not None:
        word_vectors, _ = load_word_vectors(pretrained_model, embedding_size, word_vectors_cache)
    else:
        word_vectors = None
    net = CNFNet(word_vectors=word_vectors, num_dimensions=num_dimensions, dataset=dataset, encoding_params=encoding_params, coupling_hidden_size=coupling_hidden_size, coupling_hidden_layers=coupling_hidden_layers, coupling_num_flows=coupling_num_flows, coupling_num_mixtures=coupling_num_mixtures, coupling_dropout=coupling_dropout, coupling_input_dropout=coupling_input_dropout, max_seq_len=max_seq_len, use_time_embed=use_time_embed)
    return net
