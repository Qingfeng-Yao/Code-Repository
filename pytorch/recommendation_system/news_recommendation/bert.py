import torch.nn as nn

from pytorch_pretrained_bert.modeling import BertModel


class BERT(nn.Module):
    """Class for loading pretrained BERT model."""

    def __init__(self, update_embedding=False, embedding_reduction='none', pretrained_model_name=None, cache_dir=None):
        super().__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, cache_dir=cache_dir)
        self.embedding = self.bert.embeddings
        self.embedding_size = self.embedding.word_embeddings.embedding_dim

        self.reduction = embedding_reduction

        # (Remove or not) BERT model parameters from optimization
        for param in self.bert.parameters():
            param.requires_grad = update_embedding

    def forward(self, x):
        # x.shape = (batch_size, sentence_length)

        self.bert.eval()  # make sure bert is in eval() mode
        hidden, _ = self.bert(x, output_all_encoded_layers=False)  # output only last layer
        # hidden.shape = (batch_size, sentence_length, hidden_size)

        return hidden
