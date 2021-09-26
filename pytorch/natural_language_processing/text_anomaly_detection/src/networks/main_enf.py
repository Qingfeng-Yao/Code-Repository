from .enf_Net import ENFNet
from .bert import BERT
from base.embedding import MyEmbedding
from utils.word_vectors import load_word_vectors


def build_network(net_name, dataset, embedding_size=None, pretrained_model=None, update_embedding=True,
                  embedding_reduction='none', use_tfidf_weights=False, normalize_embedding=False,
                  word_vectors_cache='../data/word_vectors_cache', coupling_hidden_size=1024, coupling_hidden_layers=2, coupling_num_flows=1, coupling_num_mixtures=64, coupling_dropout=0.0, coupling_input_dropout=0.0, max_seq_len=None, use_time_embed=False):
    """Builds the neural network."""

    net = None
    vocab_size = dataset.encoder.vocab_size

    # Set embedding

    # Load pre-trained model if specified
    if pretrained_model is not None:
        # if word vector model
        if pretrained_model in ['GloVe_6B', 'GloVe_42B', 'GloVe_840B', 'GloVe_twitter.27B', 'FastText_en']:
            word_vectors, embedding_size = load_word_vectors(pretrained_model, embedding_size, word_vectors_cache)
            embedding = MyEmbedding(vocab_size, embedding_size, update_embedding, embedding_reduction,
                                    use_tfidf_weights, normalize_embedding)
            # Init embedding with pre-trained word vectors
            for i, token in enumerate(dataset.encoder.vocab):
                embedding.weight.data[i] = word_vectors[token]
        # if language model
        if pretrained_model in ['bert']:
            embedding = BERT()
    else:
        if embedding_size is not None:
            embedding = MyEmbedding(vocab_size, embedding_size, update_embedding, embedding_reduction,
                                    use_tfidf_weights, normalize_embedding)
        else:
            raise Exception('If pretrained_model is None, embedding_size must be specified')

    net = ENFNet(embedding, coupling_hidden_size=coupling_hidden_size, coupling_hidden_layers=coupling_hidden_layers, coupling_num_flows=coupling_num_flows, coupling_num_mixtures=coupling_num_mixtures, coupling_dropout=coupling_dropout, coupling_input_dropout=coupling_input_dropout, max_seq_len=max_seq_len, use_time_embed=use_time_embed)
    return net
