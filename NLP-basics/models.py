import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel

class TextCNN(nn.Module):
    def __init__(self, args, vocab_size, num_classes):
        super(TextCNN, self).__init__()
        self.sequence_length = args.sequence_length
        self.filter_sizes = args.filter_sizes
        self.num_filters_total = args.num_filters * len(args.filter_sizes)
        self.W = nn.Embedding(vocab_size, args.embedding_size)
        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, args.num_filters, (size, args.embedding_size)) for size in args.filter_sizes])

    def forward(self, X):
        embedded_chars = self.W(X) # [batch_size, sequence_length, embedding_size]
        embedded_chars = embedded_chars.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            h = F.relu(conv(embedded_chars))
            # mp : ((filter_height, filter_width))
            mp = nn.MaxPool2d((self.sequence_length - self.filter_sizes[i] + 1, 1))
            # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(self.filter_sizes)) # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total]) # [batch_size(=6), output_height * output_width * (output_channel * 3)]
        model = self.Weight(h_pool_flat) + self.Bias # [batch_size, num_classes]
        return model

class BiLSTM_Attention(nn.Module):
    def __init__(self, args, vocab_size, num_classes, device):
        super(BiLSTM_Attention, self).__init__()
        self.n_hidden = args.n_hidden
        self.device = device
        self.embedding = nn.Embedding(vocab_size, args.embedding_size)
        self.lstm = nn.LSTM(args.embedding_size, args.n_hidden, bidirectional=True)
        self.out = nn.Linear(args.n_hidden * 2, num_classes)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.cpu().numpy() # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, X):
        input = self.embedding(X) # input : [batch_size, len_seq, embedding_dim]
        input = input.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]

        hidden_state = torch.zeros(1*2, len(X), self.n_hidden).to(self.device) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(1*2, len(X), self.n_hidden).to(self.device) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state) # attention : [batch_size, n_step]
        return self.out(attn_output) # model : [batch_size, num_classes]

class BERT(nn.Module):
    """Class for loading pretrained BERT model."""

    def __init__(self, num_classes, pretrained_model_name='bert-base-chinese', cache_dir='data/bert_cache', embedding_reduction='mean'):
        super().__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, cache_dir=cache_dir)
        self.embedding = self.bert.embeddings
        # print(self.embedding.word_embeddings.weight.data.shape) # torch.Size([21128, 768]) 
        self.embedding_size = self.embedding.word_embeddings.embedding_dim
        # print("embedding_size {}".format(self.embedding_size)) # 768

        self.reduction = embedding_reduction
        self.out = nn.Linear(self.embedding_size, num_classes)

    def forward(self, x):
        hidden, _ = self.bert(x, output_all_encoded_layers=False)  # output only last layer
        # hidden.shape = (batch_size, sentence_length, hidden_size)

        # Change to hidden.shape = (sentence_length, batch_size, hidden_size) align output with word embeddings
        embedded = hidden.transpose(0, 1)

        if self.reduction == 'mean':
            embedded = torch.mean(embedded, dim=0)
            embedded = embedded / torch.norm(embedded, p=2, dim=1, keepdim=True).clamp(min=1e-08)
            embedded[torch.isnan(embedded)] = 0

        return self.out(embedded)