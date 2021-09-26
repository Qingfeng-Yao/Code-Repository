from networks.main_cnf import build_network
from base.base_dataset import BaseADDataset
from optim.cnf_trainer import CNFTrainer

import json

class CNF(object):
    """A class for CNF models."""

    def __init__(self):
        """Init CNF instance."""

        # CNF network: encoding module + normalization flow module
        self.net_name = None
        self.net = None

        self.trainer = None
        self.optimizer_name = None

    def set_network(self, net_name, dataset, pretrained_model, embedding_size, num_dimensions=None, encoding_params=None, coupling_hidden_size=None, coupling_hidden_layers=None, coupling_num_flows=None, coupling_num_mixtures=None, coupling_dropout=None, coupling_input_dropout=None, max_seq_len=None, use_time_embed=None):
        """Builds the CNF network composed of a encoding module and a normalization flow module."""
        self.net_name = net_name
        self.net = build_network(net_name, dataset, pretrained_model=pretrained_model, embedding_size=embedding_size, num_dimensions=num_dimensions, encoding_params=encoding_params, coupling_hidden_size=coupling_hidden_size, coupling_hidden_layers=coupling_hidden_layers, coupling_num_flows=coupling_num_flows, coupling_num_mixtures=coupling_num_mixtures, coupling_dropout=coupling_dropout, coupling_input_dropout=coupling_input_dropout, max_seq_len=max_seq_len, use_time_embed=use_time_embed)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 25,
              lr_milestones: tuple = (), batch_size: int = 64, use_length_prior: bool = False, prior_dist_params: dict = {}, weight_decay: float = 0.5e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the EmbeddingNF model on the training data."""
        self.optimizer_name = optimizer_name
        self.trainer = CNFTrainer(optimizer_name, lr, n_epochs, lr_milestones, batch_size, use_length_prior, prior_dist_params, weight_decay, device, n_jobs_dataloader)
        self.net = self.trainer.train(dataset, self.net)

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the EmbeddingNF model on the test data."""
        if self.trainer is None:
            self.trainer = CNFTrainer(device, n_jobs_dataloader)

        self.trainer.test(dataset, self.net)

    def save_model(self, export_path):
        """Save EmbeddingNF model to export_path."""
        # TODO: Implement save_model
        pass

    def load_model(self, import_path, device: str = 'cuda'):
        """Load EmbeddingNF model from import_path."""
        # TODO: Implement load_model
        pass

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
