from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from networks.enf_Net import ENFNet
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np

class ENFTrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)
        self.test_auc = 0.0
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: ENFNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set parameters and optimizer (Adam optimizer for now)
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                _, length_batch, text_batch, _, _ = data
                text_batch = text_batch.to(self.device)
                length_batch = length_batch.to(self.device)
                # text_batch.shape = (sentence_length, batch_size)
                # length_batch.shape = (batch_size, )

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                # forward pass
                loss, _ = net(text_batch, length_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)  # clip gradient norms in [-0.5, 0.5]
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time

        # Log results
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: ENFNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        idx_label_score = []
        start_time = time.time()
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                idx, length_batch, text_batch, label_batch, _ = data
                text_batch, length_batch, label_batch = text_batch.to(self.device), length_batch.to(self.device), label_batch.to(self.device)

                # forward pass
                loss, ad_scores = net(text_batch, length_batch)

                # Save tuples of (idx, label, score) in a list
                idx_label_score += list(zip(idx,
                                                 label_batch.cpu().data.numpy().tolist(),
                                                 ad_scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time

        # Save list of (idx, label, score) tuples
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        if np.sum(labels) > 0:
            self.test_auc = roc_auc_score(labels, scores)
        else:
            self.test_auc = 0.0

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')


