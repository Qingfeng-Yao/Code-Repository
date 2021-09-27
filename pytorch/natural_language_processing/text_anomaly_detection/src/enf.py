from networks.main_enf import build_network
from base.base_dataset import BaseADDataset
from optim.enf_trainer import ENFTrainer

import json

class EmbeddingNF(object):
    """A class for EmbeddingNF models."""

    def __init__(self):
        """Init EmbeddingNF instance."""

        # EmbeddingNF network: pretrained_model (word embedding or language model) + normalization flow module
        self.net_name = None
        self.net = None

        self.trainer = None
        self.optimizer_name = None

    def set_network(self, net_name, dataset, pretrained_model, embedding_size=None, embedding_reduction='none', flow_type=None, coupling_hidden_size=None, coupling_num_flows=None, use_length_prior=True, device='cuda'):
        """Builds the EmbeddingNF network composed of a pretrained_model and a normalization flow module."""
        self.net_name = net_name
        self.net = build_network(net_name, dataset, embedding_size=embedding_size, pretrained_model=pretrained_model, embedding_reduction=embedding_reduction, flow_type=flow_type, 
                                 update_embedding=True, coupling_hidden_size=coupling_hidden_size, coupling_num_flows=coupling_num_flows, use_length_prior=use_length_prior, device=device)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 25,
              lr_milestones: tuple = (), batch_size: int = 64, weight_decay: float = 0.5e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the EmbeddingNF model on the training data."""
        self.optimizer_name = optimizer_name
        self.trainer = ENFTrainer(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device, n_jobs_dataloader)
        self.net = self.trainer.train(dataset, self.net)

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the EmbeddingNF model on the test data."""
        if self.trainer is None:
            self.trainer = ENFTrainer(device, n_jobs_dataloader)

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
