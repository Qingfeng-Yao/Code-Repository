from .enf_Net import ENFNet
from .enf_Net_2D import ENFNet_2D
from .bert import BERT
from base.embedding import MyEmbedding
from utils.word_vectors import load_word_vectors


def build_network(net_name, dataset, embedding_size=None, pretrained_model=None, update_embedding=True,
                  embedding_reduction='none', use_tfidf_weights=False, normalize_embedding=False,
                  word_vectors_cache='../data/word_vectors_cache', flow_type=None, coupling_hidden_size=1024, coupling_num_flows=1, use_length_prior=True, device='cuda'):
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
    if embedding_reduction == 'none':
        net = ENFNet(embedding, dataset=dataset, flow_type=flow_type, coupling_hidden_size=coupling_hidden_size, coupling_num_flows=coupling_num_flows, use_length_prior=use_length_prior, device=device)
    else:
        net = ENFNet_2D(embedding, dataset=dataset, flow_type=flow_type, coupling_hidden_size=coupling_hidden_size, coupling_num_flows=coupling_num_flows, use_length_prior=use_length_prior, device=device)
    return net
