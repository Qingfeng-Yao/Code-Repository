from base.base_dataset import BaseADDataset
from networks.main_cvdd_flow import build_network
from optim.cvdd_flow_trainer import CVDDFloTrainer

import json


class CVDD_Flow(object):

    def __init__(self, ad_score='context_dist_mean'):

        # Anomaly score function
        self.ad_score = ad_score

        self.net = None

        self.trainer = None
        self.optimizer_name = None

        self.train_dists = None
        self.train_top_words = None

        self.test_dists = None
        self.test_top_words = None
        self.test_att_weights = None

        self.results = {
            'context_vectors': None,
            'train_time': None,
            'train_att_matrix': None,
            'test_time': None,
            'test_att_matrix': None,
            'test_auc': None,
            'test_scores': None
        }

    def set_network(self, net_name, dataset, pretrained_model, embedding_size=None, attention_size=150,
                    n_attention_heads=3):
        self.net_name = net_name
        self.net = build_network(net_name, dataset, embedding_size=embedding_size, pretrained_model=pretrained_model,
                                 update_embedding=False, attention_size=attention_size,
                                 n_attention_heads=n_attention_heads)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 25,
              lr_milestones: tuple = (), batch_size: int = 64, lambda_p: float = 1.0,
              alpha_scheduler: str = 'logarithmic', weight_decay: float = 0.5e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the CVDD model on the training data."""
        self.optimizer_name = optimizer_name
        self.trainer = CVDDFloTrainer(optimizer_name, lr, n_epochs, lr_milestones, batch_size, lambda_p, alpha_scheduler,
                                   weight_decay, device, n_jobs_dataloader)
        self.net = self.trainer.train(dataset, self.net)

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the CVDD model on the test data."""

        if self.trainer is None:
            self.trainer = CVDDFloTrainer(device, n_jobs_dataloader)

        self.trainer.test(dataset, self.net, ad_score=self.ad_score)

    def save_model(self, export_path):
        """Save CVDD model to export_path."""
        # TODO: Implement save_model
        pass

    def load_model(self, import_path, device: str = 'cuda'):
        """Load CVDD model from import_path."""
        # TODO: Implement load_model
        pass

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
