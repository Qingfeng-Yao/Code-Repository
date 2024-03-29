import click
import torch
import logging
import random
import numpy as np
import os
import pickle as pkl
import datetime

from utils.config import Config
from utils.distributions import PriorDistribution

from cvdd import CVDD
from cvdd_flow import CVDD_Flow
from enf import EmbeddingNF
from cnf import CNF
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['reuters', 'newsgroups20', 'imdb']))
@click.argument('net_name', type=click.Choice(['cvdd_Net', 'CNF', 'EmbeddingNF', 'date_Net', 'cvdd_flow']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--device', type=str, default='cuda:1', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--tokenizer', default='spacy', type=click.Choice(['spacy', 'bert']), help='Select text tokenizer.')
@click.option('--clean_txt', is_flag=True, help='Specify if text should be cleaned in a pre-processing step.')
@click.option('--embedding_size', type=int, default=None, help='Size of the word vector embedding.')
@click.option('--pretrained_model', default=None,
              type=click.Choice([None, 'GloVe_6B', 'GloVe_42B', 'GloVe_840B', 'GloVe_twitter.27B', 'FastText_en',
                                 'bert']),
              help='Load pre-trained word vectors or language models to initialize the word embeddings.')
@click.option('--embedding_reduction', default='none',
              type=click.Choice(['none', 'max', 'mean']),
              help='Choose embedding reduction method.')

@click.option('--num_dimensions', type=int, default=3, help='Dimensionality of the embeddings.')
@click.option('--flow_type', type=click.Choice(['maf', 'glow', 'bert-glow', 'realnvp', 'maf-split', 'maf-split-glow']), default='maf', help='Selecting the flow in the embeddingNF.')

@click.option('--coupling_hidden_size', type=int, default=1024, help='Hidden size of the coupling layers.')
@click.option('--coupling_hidden_layers', type=int, default=2, help='Number of hidden layers in the coupling layers.')
@click.option('--coupling_num_flows', type=int, default=1, help='Number of coupling layers to use.')
@click.option('--coupling_num_mixtures', type=int, default=64, help='Number of mixtures used in the coupling layers.')
@click.option('--coupling_dropout', type=float, default=0.0, help='Number of coupling layers to use.')
@click.option('--coupling_input_dropout', type=float, default=0.0, help='Number of coupling layers to use.')
@click.option('--max_seq_len', type=int, default=None, help='Maximum sequence length. reuters: 550, newsgroups20: 7337, imdb: 1400.')
@click.option('--use_length_prior', is_flag=True, help='Specify if length prior should be used in loss computation.')
@click.option('--use_time_embed', is_flag=True, help='Specify if time embeddings should be used in a lstm module.')

@click.option('--prior_dist_type', type=click.Choice([PriorDistribution.LOGISTIC, PriorDistribution.GAUSSIAN]), default=PriorDistribution.GAUSSIAN, help='Selecting the prior distribution that should be used.')
@click.option('--prior_dist_mu', type=float, default=None, help='Center location of the distribution.')
@click.option('--prior_dist_sigma', type=float, default=None, help='Scaling of the distribution.')
@click.option('--prior_dist_start_x', type=float, default=None, help='If distribution is bounded, but should be shifted, this parameter determines the start position.')
@click.option('--prior_dist_stop_x', type=float, default=None, help='If distribution is bounded, but should be shifted, this parameter determines the end position.')

@click.option('--ad_score', default='context_dist_mean', type=click.Choice(['context_dist_mean', 'context_best']),
              help='Choose the AD score function')
@click.option('--n_attention_heads', type=int, default=1, help='Number of attention heads in self-attention module.')
@click.option('--attention_size', type=int, default=100, help='Self-attention module dimensionality.')
@click.option('--lambda_p', type=float, default=1.0,
              help='Hyperparameter for context vector orthogonality regularization P = (CCT - I)')
@click.option('--alpha_scheduler', default='logarithmic', type=click.Choice(['soft', 'linear', 'logarithmic', 'hard']),
              help='Set annealing strategy for temperature hyperparameter alpha.')

@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=64, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=0.5e-6,
              help='Weight decay (L2 penalty) hyperparameter.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--n_threads', type=int, default=0,
              help='Sets the number of OpenMP threads used for parallelizing CPU operations')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, device, seed, tokenizer, clean_txt,
         embedding_size, pretrained_model, embedding_reduction, num_dimensions, flow_type, coupling_hidden_size, coupling_hidden_layers, coupling_num_flows, coupling_num_mixtures, coupling_dropout, coupling_input_dropout, max_seq_len, use_length_prior, use_time_embed, prior_dist_type, prior_dist_mu, prior_dist_sigma, prior_dist_start_x, prior_dist_stop_x, ad_score, n_attention_heads, attention_size, lambda_p, alpha_scheduler,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, n_jobs_dataloader, n_threads,
         normal_class):
    """
    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    xp_path += '/text_{}_{}'.format(dataset_name, net_name)
    if not os.path.exists(xp_path):
        os.makedirs(xp_path)
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)
    logger.info('Tokenizer: %s' % cfg.settings['tokenizer'])
    logger.info('Clean text in pre-processing: %s' % cfg.settings['clean_txt'])
    if cfg.settings['embedding_size'] is not None:
        logger.info('Word vector embedding size: %d' % cfg.settings['embedding_size'])
    logger.info('Load pre-trained model: %s' % cfg.settings['pretrained_model'])

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Set seed for reproducibility
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)
    if n_threads > 0:
        torch.set_num_threads(n_threads)
        logger.info('Number of threads used for parallelizing CPU operations: %d' % n_threads)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, cfg.settings['tokenizer'],
                           clean_txt=cfg.settings['clean_txt'], max_seq_len=cfg.settings['max_seq_len'])
    if net_name == 'CNF':
        # Initialize CNF model
        cnf = CNF()
        encoding_params = {
            "num_flows": 0,
            "hidden_layers": 2,
            "hidden_size": 128
        }
        cnf.set_network(net_name=net_name,
                        dataset=dataset,
                        pretrained_model=cfg.settings['pretrained_model'],
                        embedding_size=cfg.settings['embedding_size'],
                        num_dimensions=cfg.settings['num_dimensions'],
                        encoding_params=encoding_params,
                        coupling_hidden_size=cfg.settings['coupling_hidden_size'],
                        coupling_hidden_layers=cfg.settings['coupling_hidden_layers'],
                        coupling_num_flows=cfg.settings['coupling_num_flows'],
                        coupling_num_mixtures=cfg.settings['coupling_num_mixtures'],
                        coupling_dropout=cfg.settings['coupling_dropout'],
                        coupling_input_dropout=cfg.settings['coupling_input_dropout'],
                        max_seq_len=cfg.settings['max_seq_len'],
                        use_time_embed=cfg.settings['use_time_embed'])

        # If specified, load model parameters from already trained model
        if load_model:
            cnf.load_model(import_path=load_model, device=device)
            logger.info('Loading model from %s.' % load_model)

        # Train model on dataset
        prior_dist_params = {
        "distribution_type": cfg.settings['prior_dist_type'],
        "mu": cfg.settings['prior_dist_mu'],
        "sigma": cfg.settings['prior_dist_sigma'],
        "start_x": cfg.settings['prior_dist_start_x'],
        "stop_x": cfg.settings['prior_dist_stop_x']
        }
        cnf.train(dataset,
                optimizer_name=cfg.settings['optimizer_name'],
                lr=cfg.settings['lr'],
                n_epochs=cfg.settings['n_epochs'],
                lr_milestones=cfg.settings['lr_milestone'],
                batch_size=cfg.settings['batch_size'],
                use_length_prior=cfg.settings['use_length_prior'],
                prior_dist_params=prior_dist_params,
                weight_decay=cfg.settings['weight_decay'],
                device=device,
                n_jobs_dataloader=n_jobs_dataloader)

        # Test model
        cnf.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)
    elif net_name == 'EmbeddingNF':
        # Initialize EmbeddingNF model and set word embedding
        enf = EmbeddingNF()
        enf.set_network(net_name=net_name,
                        dataset=dataset,
                        pretrained_model=cfg.settings['pretrained_model'],
                        embedding_size=cfg.settings['embedding_size'],
                        embedding_reduction=cfg.settings['embedding_reduction'],
                        flow_type=cfg.settings['flow_type'],
                        coupling_hidden_size=cfg.settings['coupling_hidden_size'],
                        coupling_num_flows=cfg.settings['coupling_num_flows'],
                        use_length_prior=cfg.settings['use_length_prior'],
                        device=cfg.settings['device'])

        # If specified, load model parameters from already trained model
        if load_model:
            enf.load_model(import_path=load_model, device=device)
            logger.info('Loading model from %s.' % load_model)

        # Train model on dataset
        enf.train(dataset,
                optimizer_name=cfg.settings['optimizer_name'],
                lr=cfg.settings['lr'],
                n_epochs=cfg.settings['n_epochs'],
                lr_milestones=cfg.settings['lr_milestone'],
                batch_size=cfg.settings['batch_size'],
                weight_decay=cfg.settings['weight_decay'],
                device=device,
                n_jobs_dataloader=n_jobs_dataloader)

        # Test model
        enf.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)
    elif net_name == 'date_Net':
        os.environ["TOKENIZERS_PARALLELISM"] = 'false'
        masks_ = pkl.load(open(data_path+'/pseudo_labels128_p50.pkl', 'rb'))
        subset = [dataset.subset]
        tensorboard_dir = 'run'
        exp_prefix = 'tests'
        now = datetime.now()
        date_time = now.strftime("%m%d%Y_%H%M%S")
        run_name = f'{subset[0]}_{date_time}'

        train_args = {
        "fp16": False,
        "use_multiprocessing": False,
        "reprocess_input_data": False,
        "overwrite_output_dir": True,
        "num_train_epochs": 20,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "learning_rate": 1e-5,
        "warmup_steps": 1000,
        "train_batch_size": 16,  #was 32
        "eval_batch_size": 16,  #was 32
        "gradient_accumulation_steps": 1,
        "block_size": 128 + 2,
        "max_seq_length": 128 + 2,
        "dataset_type": "simple",
        "logging_steps": 500,
        "evaluate_during_training": True,
        "evaluate_during_training_steps": 500,  #was 500
        "evaluate_during_training_steps_anomaly": 500,  #was 500
        "anomaly_batch_size": 16,
        "evaluate_during_training_verbose": True,
        "use_cached_eval_features": True,
        "sliding_window": True,
        "vocab_size": 52000,
        "eval_anomalies": True,
        "random_generator": 1,
        "use_rtd_loss": True,
        "rtd_loss_weight": 50,
        "rmd_loss_weight": 100,
        "mlm_loss_weight": 1,
        "dump_histogram": 0,
        "eval_anomaly_after": 0,
        "train_just_generator": 0,
        "replace_tokens": 0,
        "extract_scores": 1,
        "subset_name": subset[0],
        "extract_repr": 0,
        # "vanilla_electra": {
        #     "no_masks": masks,
        # },
        # "vanilla_electra": False,
        "train_document": True,
        "tokenizer_name": "bert-base-uncased",
        "tensorboard_dir": f'{tensorboard_dir}/{exp_prefix}/{run_name}',
        "extract_reps": 0,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "scores_export_path": f"./token_scores/{run_name}/",
        "generator_config": {
            "embedding_size": 128,
            "hidden_size": 16,
            "num_hidden_layers": 1,
        },
        "discriminator_config": {
            "hidden_dropout_prob": 0.5,
            "attention_probs_dropout_prob": 0.5,
            "embedding_size": 128,
            "hidden_size": 256,
            "num_hidden_layers": 4,
        },
        "mlm_lr_ratio": 1,
        }

        train_file = f"{data_path}/{dataset_name}/train/{subset[0]}.txt"
        test_file = f"{data_path}/{dataset_name}/test/{subset[0]}.txt"

        outlier_file = f"{data_path}/{dataset_name}/test/{subset[0]}-outliers.txt"

    elif net_name == 'cvdd_Net':
        # Print CVDD configuration
        logger.info('Anomaly Score: %s' % cfg.settings['ad_score'])
        logger.info('Number of attention heads: %d' % cfg.settings['n_attention_heads'])
        logger.info('Attention size: %d' % cfg.settings['attention_size'])
        logger.info('Orthogonality regularization hyperparameter: %.3f' % cfg.settings['lambda_p'])
        logger.info('Temperature alpha annealing strategy: %s' % cfg.settings['alpha_scheduler'])

        # Initialize CVDD model and set word embedding
        cvdd = CVDD(cfg.settings['ad_score'])
        cvdd.set_network(net_name=net_name,
                        dataset=dataset,
                        pretrained_model=cfg.settings['pretrained_model'],
                        embedding_size=cfg.settings['embedding_size'],
                        attention_size=cfg.settings['attention_size'],
                        n_attention_heads=cfg.settings['n_attention_heads'])

        # If specified, load model parameters from already trained model
        if load_model:
            cvdd.load_model(import_path=load_model, device=device)
            logger.info('Loading model from %s.' % load_model)

        # Train model on dataset
        cvdd.train(dataset,
                optimizer_name=cfg.settings['optimizer_name'],
                lr=cfg.settings['lr'],
                n_epochs=cfg.settings['n_epochs'],
                lr_milestones=cfg.settings['lr_milestone'],
                batch_size=cfg.settings['batch_size'],
                lambda_p=cfg.settings['lambda_p'],
                alpha_scheduler=cfg.settings['alpha_scheduler'],
                weight_decay=cfg.settings['weight_decay'],
                device=device,
                n_jobs_dataloader=n_jobs_dataloader)

        # Test model
        cvdd.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)
    elif net_name == 'cvdd_flow':
        # Initialize CVDD_Flow model and set word embedding
        cvdd_flow = CVDD_Flow(cfg.settings['ad_score'])
        cvdd_flow.set_network(net_name=net_name,
                        dataset=dataset,
                        pretrained_model=cfg.settings['pretrained_model'],
                        embedding_size=cfg.settings['embedding_size'],
                        attention_size=cfg.settings['attention_size'],
                        n_attention_heads=cfg.settings['n_attention_heads'])
        # Train model on dataset
        cvdd_flow.train(dataset,
                optimizer_name=cfg.settings['optimizer_name'],
                lr=cfg.settings['lr'],
                n_epochs=cfg.settings['n_epochs'],
                lr_milestones=cfg.settings['lr_milestone'],
                batch_size=cfg.settings['batch_size'],
                lambda_p=cfg.settings['lambda_p'],
                alpha_scheduler=cfg.settings['alpha_scheduler'],
                weight_decay=cfg.settings['weight_decay'],
                device=device,
                n_jobs_dataloader=n_jobs_dataloader)

        # Test model
        cvdd_flow.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)


if __name__ == '__main__':
    main()
