import math
import time
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import mrr_score, ndcg_score
from bert import BERT
from mimn import MIMNCell

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, inputsize):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.output_dim = num_heads * head_size

        self.wq = nn.Linear(inputsize, self.output_dim, bias=False)
        self.wk = nn.Linear(inputsize, self.output_dim, bias=False)
        self.wv = nn.Linear(inputsize, self.output_dim, bias=False)


    def split_heads(self, x):
        x = x.view((-1, x.size(1), self.num_heads, self.head_size))
        x = x.permute([0, 2, 1, 3])
        return x

    def forward(self, key, key_mask, query, value, no_multi_head=False, return_w=False):
        # key: batch_size*(negnums+1), title_size, word_embed_size
        # key_mask: batch_size*(negnums+1), 1, 1, title_size
        q = self.wq(query) # batch_size*(negnums+1), title_size, num_heads*head_size
        k = self.wk(key) # batch_size*(negnums+1), title_size, num_heads*head_size
        v = self.wv(value) # batch_size*(negnums+1), title_size, num_heads*head_size
        if not no_multi_head:
            q = self.split_heads(q) # batch_size*(negnums+1), num_heads, title_size, head_size
            k = self.split_heads(k) # batch_size*(negnums+1), num_heads, title_size, head_size
            v = self.split_heads(v) # batch_size*(negnums+1), num_heads, title_size, head_size
            matmul_qk = torch.matmul(q, k.permute([0, 1, 3, 2]))  # batch_size*(negnums+1), num_heads, title_size, title_size
            scaled_attention_logits = matmul_qk / math.sqrt(k.size(-1))
            if key_mask is not None:
                scaled_attention_logits += key_mask
            attention_weight = F.softmax(scaled_attention_logits, dim=3) # batch_size*(negnums+1), num_heads, title_size, title_size
            output = torch.matmul(attention_weight, v) # batch_size*(negnums+1), num_heads, title_size, head_size
            output = output.permute([0, 2, 1, 3]) # batch_size*(negnums+1), title_size, num_heads, head_size
            output = output.contiguous().view((-1, output.size(1), self.output_dim)) # batch_size*(negnums+1), title_size, num_heads*head_size
        else:
            matmul_qk = torch.matmul(q, k.permute([0, 2, 1]))  # batch_size*(negnums+1), title_size, title_size
            scaled_attention_logits = matmul_qk / math.sqrt(k.size(-1))
            if key_mask is not None:
                scaled_attention_logits += torch.squeeze(key_mask, dim=1)
            attention_weight = F.softmax(scaled_attention_logits, dim=2) # batch_size*(negnums+1), title_size, title_size
            output = torch.matmul(attention_weight, v) # batch_size*(negnums+1), title_size, num_heads*head_size

        if return_w:
            return output, attention_weight
        else:
            return output

class AdditiveAttention(nn.Module):
    ''' AttentionPooling used to weighted aggregate news vectors
    Arg: 
        d_h: the last dimension of input
    '''
    def __init__(self, d_h, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, mask=None, query=None, need_dim=1, is_pooling=True, re_w=False):
        if query is not None:
            e = self.att_fc1(query)
            e = nn.Tanh()(e)
            alpha = self.att_fc2(e)
            if mask is not None:
                alpha += mask
            attention_weight = F.softmax(alpha, need_dim)
        else:
            e = self.att_fc1(x)
            e = nn.Tanh()(e)
            alpha = self.att_fc2(e)
            if mask is not None:
                alpha += mask
            attention_weight = F.softmax(alpha, need_dim)
            
        if is_pooling:
            if need_dim != 1:
                output = torch.matmul(torch.squeeze(attention_weight, -1), x)
            else:
                output = torch.sum(attention_weight * x, need_dim)
        else:
            output = attention_weight * x

        if re_w:
            return output, attention_weight
        else:
            return output

class TitleLayer(nn.Module):
    def __init__(self, word_dict, embeddings_matrix, args):
        super(TitleLayer, self).__init__()
        self.output_dim = args.num_heads * args.head_size
        self.medialayer = args.medialayer
        if (embeddings_matrix is not None):
            self.embedding = nn.Embedding(len(word_dict), args.word_embed_size)
            for i, token in enumerate(word_dict.keys()):
                self.embedding.weight.data[i] = embeddings_matrix[token]
        else:
            if args.pretrained_embeddings == 'bert':
                if args.dataset == 'heybox':
                    self.embedding = BERT(pretrained_model_name='data/ch_bert_cache', cache_dir='data/ch_bert_cache')
                elif args.dataset == 'MIND':
                    self.embedding = BERT(pretrained_model_name='data/en_bert_cache', cache_dir='data/en_bert_cache')
            else:
                self.embedding = nn.Embedding(len(word_dict), args.word_embed_size)
        self.dropout1 = nn.Dropout(args.droprate)
        self.dropout2 = nn.Dropout(args.droprate)
        self.multiatt = MultiHeadAttention(args.num_heads, args.head_size, args.word_embed_size)
        self.addatt = AdditiveAttention(self.output_dim, self.medialayer)
        self.device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')


    def forward(self, newstitle): # batch_size*(negnums+1), title_size
        x = self.embedding(newstitle) # batch_size*(negnums+1), title_size, word_embed_size
        x = self.dropout1(x)
        mask = torch.eq(newstitle, 0).float().to(self.device) # batch_size*(negnums+1), title_size
        mask = mask.masked_fill(mask == 1, -1e9)
        mask1 = torch.unsqueeze(torch.unsqueeze(mask, 1), 1) # batch_size*(negnums+1), 1, 1, title_size
        
        selfattn_output = self.multiatt(x, mask1, x, x) # batch_size*(negnums+1), title_size, num_heads*head_size
        selfattn_output = self.dropout2(selfattn_output)

        mask2 = torch.unsqueeze(mask, 2) # batch_size*(negnums+1), title_size, 1
        output = self.addatt(selfattn_output, mask2) # batch_size*(negnums+1), num_heads*head_size
        return output

class IDemblayer(nn.Module):
    def __init__(self, num_categories, args, text='category'):
        super(IDemblayer, self).__init__()
        self.embedding = nn.Embedding(num_categories, args.categ_embed_size)
        if text == 'category':
            if args.dataset == 'MIND':
                self.output_dim = args.categ_embed_size * 2
            elif args.dataset == 'heybox':
                self.output_dim = args.categ_embed_size
        else:
            self.output_dim = args.categ_embed_size

    def forward(self, inputs): # batch_size*(negnums+1), 2
        catedembed = self.embedding(inputs) # batch_size*(negnums+1), 2, categ_embed_size
        output = catedembed.view((-1, self.output_dim)) # batch_size*(negnums+1), 2*categ_embed_size
        return output

class NewsEncoder(nn.Module):
    def __init__(self, word_dict, preembed, categories, users, num_ctr_bin, args):
        super(NewsEncoder, self).__init__()
        self.titlelayer = TitleLayer(word_dict, preembed, args)
        self.categlayer = IDemblayer(categories, args, text='category')
        self.ctrlayer = IDemblayer(num_ctr_bin, args, text='ctr')
        if users is not None:
            self.userlayer = IDemblayer(users, args, text='user')
        else:
            self.userlayer = None
        self.title_size = args.title_size

        self.use_topic = args.use_topic

    def forward(self, inputs, return_ctr=False): # batch_size*(negnums+1), title_size+3
        title_embed = self.titlelayer(inputs[:, :self.title_size]) # batch_size*(negnums+1), num_heads*head_size
        if self.use_topic:
            news_embed = title_embed
        else:
            if self.userlayer is not None:
                user_embed = self.userlayer(inputs[:, self.title_size: self.title_size+1]) # batch_size*(negnums+1), categ_embed_size
                categ_embed = self.categlayer(inputs[:, self.title_size+1:self.title_size+2]) # batch_size*(negnums+1), categ_embed_size
                news_embed = torch.cat((title_embed, user_embed, categ_embed), 1) # batch_size*(negnums+1), num_heads*head_size+2*categ_embed_size
            else:
                categ_embed = self.categlayer(inputs[:, self.title_size:self.title_size+2]) # batch_size*(negnums+1), 2*categ_embed_size
                news_embed = torch.cat((title_embed, categ_embed), 1) # batch_size*(negnums+1), num_heads*head_size+2*categ_embed_size
        ctr_emb = self.ctrlayer(inputs[:, self.title_size+2:]) # batch_size*(negnums+1), categ_embed_size
        if return_ctr:
            return news_embed, ctr_emb
        else:
            return news_embed

class UserEncoder(nn.Module):
    def __init__(self, newscncoder, args):
        super(UserEncoder, self).__init__()
        self.dropout1 = nn.Dropout(args.droprate)
        self.dropout2 = nn.Dropout(args.droprate)
        if args.use_topic:
            self.newssize = args.num_heads * args.head_size + args.categ_embed_size * 2
        else:
            self.newssize = args.num_heads * args.head_size
        self.multiatt = MultiHeadAttention(args.num_heads, self.newssize // args.num_heads, self.newssize)
        self.target_att = MultiHeadAttention(args.num_heads, self.newssize // args.num_heads, self.newssize)
        self.selfaddatt = AdditiveAttention(self.newssize, args.medialayer)
        self.ctraddatt = AdditiveAttention(self.newssize+args.categ_embed_size, args.medialayer)
        self.mergeaddatt = AdditiveAttention(2*self.newssize+args.categ_embed_size, args.medialayer)
        
        self.newscncoder = newscncoder
        self.his_size = args.his_size
        self.device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')

        self.score_add = args.score_add

        self.add_op = args.add_op
        self.mean_op = args.mean_op
        self.max_op = args.max_op
        self.atten_op = args.atten_op
        self.mvke = args.mvke
        self.cross_gate = args.cross_gate

        self.dnn = args.dnn 
        self.moe = args.moe
        
        self.cross_atten = args.cross_atten
        self.cross_atten_deep = args.cross_atten_deep

        self.mimn_flag = args.mimn
        self.batch_size_ = args.batch_size
        self.max_len = args.max_len
        self.mask_flag = args.mask_flag
        self.util_reg = args.util_reg
        if self.mimn_flag: # memory_vector_dim=newssize
            self.mimn_net = MIMNCell(args.gpu, args.controller_units, args.memory_size, self.newssize, args.read_head_num, args.write_head_num, self.newssize,
                 output_dim=None, clip_value=20, shift_range=1, batch_size=args.batch_size, mem_induction=args.mem_induction, util_reg=args.util_reg, sharp_value=2.)

        self.use_ctr = args.use_ctr
        if self.use_ctr:
            self.tar_ctr_dense = nn.Linear(2*self.newssize+args.categ_embed_size, 1)
        self.merge_ctr_tar = args.merge_ctr_tar

    # for mimn net
    def clear_mask_state(self, state, begin_state, mask, t, batch_size_):
        state["controller_state"] = (1-torch.reshape(mask[:,t], (batch_size_, 1))) * begin_state["controller_state"] + torch.reshape(mask[:,t], (batch_size_, 1)) * state["controller_state"]
        state["M"] = (1-torch.reshape(mask[:,t], (batch_size_, 1, 1))) * begin_state["M"] + torch.reshape(mask[:,t], (batch_size_, 1, 1)) * state["M"]
        state["key_M"] = (1-torch.reshape(mask[:,t], (batch_size_, 1, 1))) * begin_state["key_M"] + torch.reshape(mask[:,t], (batch_size_, 1, 1)) * state["key_M"]
        state["sum_aggre"] = (1-torch.reshape(mask[:,t], (batch_size_, 1, 1))) * begin_state["sum_aggre"] + torch.reshape(mask[:,t], (batch_size_, 1, 1)) * state["sum_aggre"]

        return state


    def forward(self, user_click, seq_len, target=None, train_test_flag=0, use_ctr=False):
        # batch_size, his_size, title_size+3; batch_size, ; batch_size, negnums+1, num_heads*head_size+2*categ_embed_size
        reshape_user_click = user_click.view(-1, user_click.size(-1))
        if use_ctr:
            reshape_click_embed, reshape_click_ctr_embed = self.newscncoder(reshape_user_click, True) # batch_size*his_size, num_heads*head_size+2*categ_embed_size; batch_size*his_size, categ_embed_size
        else:
            reshape_click_embed = self.newscncoder(reshape_user_click) # batch_size*his_size, num_heads*head_size+2*categ_embed_size
        click_embed = reshape_click_embed.view(user_click.size(0), -1, reshape_click_embed.size(-1)) # batch_size, his_size, num_heads*head_size+2*categ_embed_size
        click_embed = self.dropout1(click_embed)

        if use_ctr:
            click_ctr_embed = reshape_click_ctr_embed.view(user_click.size(0), -1, reshape_click_ctr_embed.size(-1)) # batch_size, his_size, categ_embed_size

        mask = torch.arange(0, self.his_size).to(self.device).unsqueeze(0).expand(user_click.size(0), self.his_size).lt(
            seq_len.unsqueeze(1)).float() # batch_size, his_size
        
        if self.mimn_flag:
            if train_test_flag == 0:
                bz = self.batch_size_
                state = self.mimn_net.zero_state(bz)
            elif train_test_flag == 1:
                bz = 1
                state = self.mimn_net.zero_state(bz)
            begin_state = state
            self.state_list = [state]

            # self.mimn_o = []
            for t in range(self.max_len):
                state = self.mimn_net(click_embed[:, t, :], state)
                if self.mask_flag:
                    state = self.clear_mask_state(state, begin_state, mask, t, bz)
                # self.mimn_o.append(output)
                self.state_list.append(state)

            # self.mimn_o = torch.stack(self.mimn_o, dim=1)
            self.state_list.append(state)
            mean_memory = torch.mean(state['sum_aggre'], dim=-2)

            before_aggre = state['w_aggre']
            M = state['M']
            # user_embed = torch.mean(M, 1)
            if self.util_reg:
                reg_loss = self.mimn_net.capacity_loss(before_aggre)
                output = [M, reg_loss]
            output = [M]
        else:
            mask = mask.masked_fill(mask == 0, -1e9)
            mask1 = torch.unsqueeze(torch.unsqueeze(mask, 1), 1) # batch_size, 1, 1, his_size

            selfattn_output = self.multiatt(click_embed, mask1, click_embed, click_embed) # batch_size, his_size, num_heads*head_size+2*categ_embed_size
            selfattn_output = self.dropout2(selfattn_output)

            if not (self.add_op or self.mean_op or self.max_op or self.atten_op or self.dnn or self.score_add or self.moe or self.mvke or self.cross_gate):
                if target is not None:
                    if self.cross_atten:
                        _, ta_w = self.target_att(click_embed, mask1, target, click_embed, no_multi_head=True, return_w=True) # batch_size, negnums+1, his_size
                        if self.cross_atten_deep:
                            mask2 = torch.unsqueeze(mask, 2) # batch_size, his_size, 1
                            output = self.selfaddatt(selfattn_output, mask2, is_pooling=False) 
                            output = torch.matmul(ta_w, output)
                        else:
                            output = torch.matmul(ta_w, selfattn_output)
                    elif use_ctr:
                        _, ta_w = self.target_att(click_embed, mask1, target, click_embed, no_multi_head=True, return_w=True) # batch_size, negnums+1, his_size
                        b_size, cand_num, his_len = ta_w.size(0), ta_w.size(1), ta_w.size(2)
                        mask2 = torch.unsqueeze(mask, 2) # batch_size, his_size, 1
                        _, ctr_w = self.ctraddatt(selfattn_output, mask=mask2, query=torch.cat([click_ctr_embed, selfattn_output], dim=2), re_w=True) # batch_size, his_size, 1
                        ctr_w = torch.unsqueeze(torch.squeeze(ctr_w, 2), 1).expand(b_size, cand_num, his_len)

                        ctr_emb_dim = click_ctr_embed.size(-1)
                        content_emb_dim = selfattn_output.size(-1)
                        target_emb_dim = target.size(-1)
                        click_ctr_embed_to_concat = torch.unsqueeze(click_ctr_embed, 1).expand(b_size, cand_num, his_len, ctr_emb_dim)
                        selfattn_output_to_concat = torch.unsqueeze(selfattn_output, 1).expand(b_size, cand_num, his_len, content_emb_dim)
                        target_to_concat = torch.unsqueeze(target, 2).expand(b_size, cand_num, his_len, target_emb_dim)

                        if self.merge_ctr_tar:
                            mask3 = torch.unsqueeze(torch.unsqueeze(mask, 2), 1) # batch_size, 1, his_size, 1
                            output = self.mergeaddatt(selfattn_output, mask=mask3, query=torch.cat([click_ctr_embed_to_concat, selfattn_output_to_concat, target_to_concat], dim=3), need_dim=2)
                        else:
                            eta = torch.squeeze(torch.sigmoid(self.tar_ctr_dense(torch.cat([click_ctr_embed_to_concat, selfattn_output_to_concat, target_to_concat], dim=3))), 3)
                            merge_w = eta*ta_w + (1-eta)*ctr_w
                            output = torch.matmul(merge_w, selfattn_output)
                    else:
                        output = self.target_att(click_embed, mask1, target, click_embed) # batch_size, negnums+1, num_heads*head_size+2*categ_embed_size
                elif use_ctr:
                    mask2 = torch.unsqueeze(mask, 2) # batch_size, his_size, 1
                    output = self.ctraddatt(selfattn_output, mask=mask2, query=torch.cat([selfattn_output, click_ctr_embed], dim=2)) # batch_size, num_heads*head_size+2*categ_embed_size
                else:
                    mask2 = torch.unsqueeze(mask, 2) # batch_size, his_size, 1
                    output = self.selfaddatt(selfattn_output, mask2) # batch_size, num_heads*head_size+2*categ_embed_size
            else:
                output1 = self.target_att(click_embed, mask1, target, click_embed)
                mask2 = torch.unsqueeze(mask, 2) # batch_size, his_size, 1
                output2 = self.ctraddatt(selfattn_output, mask=mask2, query=torch.cat([selfattn_output, click_ctr_embed], dim=2))

                output = (output1, output2)


        return output

class CoEncoder(nn.Module):
    def __init__(self, word_dict, embeddings_matrix, args):
        super(CoEncoder, self).__init__()
        if (embeddings_matrix is not None):
            self.embedding = nn.Embedding(len(word_dict), args.word_embed_size)
            for i, token in enumerate(word_dict.keys()):
                self.embedding.weight.data[i] = embeddings_matrix[token]
        else:
            self.embedding = nn.Embedding(len(word_dict), args.word_embed_size)
        self.output_dim = args.num_heads * args.head_size
        self.dropout1 = nn.Dropout(args.droprate)
        self.dropout2 = nn.Dropout(args.droprate)
        self.multiatt = MultiHeadAttention(args.num_heads, args.head_size, args.word_embed_size)
        self.device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
        

    def forward(self, candidate_news, clicked_news):
        # batch_size, negnums+1, title_size+3; batch_size, his_size, title_size+3
        reshape_candidate_news = candidate_news.view(-1, candidate_news.size(-1))
        reshape_clicked_news = clicked_news.view(-1, clicked_news.size(-1))



class nrms(nn.Module):
    def __init__(self, word_dict, preembed, categories, users, log_users, num_ctr_bin, args):
        super(nrms, self).__init__()
        # make news and user's embedding dim the same
        # training sample and testing sample have different constructions: negnums+1, 1
        # consider two user representations: self attention and target attention
        self.newsencoder = NewsEncoder(word_dict, preembed, categories, users, num_ctr_bin, args)
        self.userencoder = UserEncoder(self.newsencoder, args)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
        self.newssize = args.num_heads * args.head_size + args.categ_embed_size * 2


        self.final_score_gate = nn.Linear(self.newssize, 1)

        self.use_ctr = args.use_ctr
        self.score_gate = args.score_gate
        self.user_ctr_dense = nn.Linear(self.newssize, 1)

        self.din = args.din # target attention
        self.cross_atten = args.cross_atten 

        self.score_add = args.score_add

        self.add_op = args.add_op
        self.mean_op = args.mean_op
        self.max_op = args.max_op
        self.atten_op = args.atten_op
        if self.atten_op:
            self.atten_op_addattr = AdditiveAttention(self.newssize, args.medialayer)

        self.dnn = args.dnn # single expert
        if self.dnn:
            self.dnn_dense_1 = nn.Linear(self.newssize*2, args.medialayer)
            self.dnn_dense_2 = nn.Linear(args.medialayer, self.newssize)

        self.moe = args.moe # multiple experts
        self.bias = args.bias
        if self.moe:
            self.expert_nets = []
            for n in range(args.num_experts):
                expert_net_1 = nn.Linear(self.newssize*2, args.medialayer).to(self.device)
                expert_net_2 = nn.Linear(args.medialayer, self.newssize).to(self.device)
                self.expert_nets.append([expert_net_1, expert_net_2])
            self.gate_net_1 = nn.Linear(self.newssize*2, args.medialayer)
            self.gate_net_2 = nn.Linear(args.medialayer, args.num_experts)
            self.user_layer = IDemblayer(log_users, args, text='user')
            if self.bias:
                self.bias_dense_1 = nn.Linear(self.newssize*2, args.medialayer)
                self.bias_dense_2 = nn.Linear(args.medialayer, self.newssize)
        self.mvke = args.mvke # multiple virtual kernel experts
        if self.mvke:
            self.user_qs = nn.Parameter(torch.ones(args.num_experts, self.newssize))
            self.expert_nets = []
            self.qs_atten_nets = []
            for n in range(args.num_experts):
                self.qs_atten_nets.append(MultiHeadAttention(args.num_heads, self.newssize // args.num_heads, self.newssize).to(self.device))
                expert_net_1 = nn.Linear(self.newssize*2, args.medialayer).to(self.device)
                expert_net_2 = nn.Linear(args.medialayer, self.newssize).to(self.device)
                self.expert_nets.append([expert_net_1, expert_net_2])
            self.gate_net = MultiHeadAttention(args.num_heads, self.newssize // args.num_heads, self.newssize)

        self.cross_gate = args.cross_gate
        if self.cross_gate:
            self.cross_gate_dense = nn.Linear(self.newssize*2, 1)


        self.mimn_flag = args.mimn
        self.util_reg = args.util_reg

        self.co_attention = args.co_attention
        if self.co_attention:
            self.co_encoder = CoEncoder(self.newsencoder, args)

    def forward(self, candidate_news, candidate_news_ctr, clicked_news, click_len, use_id, labels=None, train_test_flag=0):
        if self.co_attention:
            self.co_encoder(candidate_news, clicked_news)
        reshape_candidate_news = candidate_news.view(-1, candidate_news.size(-1)) # batch_size*(negnums+1), title_size+3
        reshape_news_embed = self.newsencoder(reshape_candidate_news) # batch_size*(negnums+1), num_heads*head_size+2*categ_embed_size
        news_embed = reshape_news_embed.view(candidate_news.size(0), -1, reshape_news_embed.size(-1)) # batch_size, negnums+1, num_heads*head_size+2*categ_embed_size
        
        if self.din or self.cross_atten:
            target = news_embed
        else:
            target = None
        user_embed = self.userencoder(clicked_news, click_len, target, train_test_flag=train_test_flag, use_ctr=self.use_ctr) 
        if self.mimn_flag:
            M = user_embed[0]
            user_final_embed = torch.mean(M, 1)
            user_final_embed = torch.unsqueeze(user_final_embed, 2)
            score = torch.squeeze(torch.matmul(news_embed, user_final_embed))
        elif self.score_add:
            user_target_embed, user_self_embed = user_embed
            b_size, l_size, e_size = user_self_embed.size(0), user_target_embed.size(1), user_self_embed.size(1)
            user_self_embed = torch.unsqueeze(user_self_embed, 1).expand(b_size, l_size, e_size)
            
            score_sa = torch.matmul(news_embed, user_self_embed.permute([0, 2, 1])) # batch_size, negnums+1, negnums+1
            score_sa = torch.squeeze(torch.diagonal(score_sa, dim1=-2, dim2=-1)) # batch_size, negnums+1
            score_ta = torch.matmul(news_embed, user_target_embed.permute([0, 2, 1])) # batch_size, negnums+1, negnums+1
            score_ta = torch.squeeze(torch.diagonal(score_ta, dim1=-2, dim2=-1)) # batch_size, negnums+1
            candidate_news_ctr = torch.squeeze(candidate_news_ctr)

            score = score_sa+score_ta+candidate_news_ctr
            
        elif self.add_op:
            user_target_embed, user_self_embed = user_embed
            b_size, l_size, e_size = user_self_embed.size(0), user_target_embed.size(1), user_self_embed.size(1)
            user_self_embed = torch.unsqueeze(user_self_embed, 1).expand(b_size, l_size, e_size)
            user_final_embed = user_target_embed+user_self_embed
            score = torch.matmul(news_embed, user_final_embed.permute([0, 2, 1])) # batch_size, negnums+1, negnums+1
            score_match = torch.squeeze(torch.diagonal(score, dim1=-2, dim2=-1)) # batch_size, negnums+1
            candidate_news_ctr = torch.squeeze(candidate_news_ctr)
            eta = torch.squeeze(torch.sigmoid(self.final_score_gate(user_final_embed)))
            score = (1-eta)*score_match+eta*candidate_news_ctr
        elif self.mean_op:
            user_target_embed, user_self_embed = user_embed
            b_size, l_size, e_size = user_self_embed.size(0), user_target_embed.size(1), user_self_embed.size(1)
            user_self_embed = torch.unsqueeze(user_self_embed, 1).expand(b_size, l_size, e_size)
            user_stack_embed = torch.stack([user_target_embed, user_self_embed], dim=2)
            user_final_embed = torch.mean(user_stack_embed, dim=2)
            score = torch.matmul(news_embed, user_final_embed.permute([0, 2, 1])) # batch_size, negnums+1, negnums+1
            score_match = torch.squeeze(torch.diagonal(score, dim1=-2, dim2=-1)) # batch_size, negnums+1
            candidate_news_ctr = torch.squeeze(candidate_news_ctr)
            eta = torch.squeeze(torch.sigmoid(self.final_score_gate(user_final_embed)))
            score = (1-eta)*score_match+eta*candidate_news_ctr
        elif self.max_op:
            user_target_embed, user_self_embed = user_embed
            b_size, l_size, e_size = user_self_embed.size(0), user_target_embed.size(1), user_self_embed.size(1)
            user_self_embed = torch.unsqueeze(user_self_embed, 1).expand(b_size, l_size, e_size)
            user_stack_embed = torch.stack([user_target_embed, user_self_embed], dim=2)
            user_final_embed, _ = torch.max(user_stack_embed, dim=2)
            score = torch.matmul(news_embed, user_final_embed.permute([0, 2, 1])) # batch_size, negnums+1, negnums+1
            score_match = torch.squeeze(torch.diagonal(score, dim1=-2, dim2=-1)) # batch_size, negnums+1
            candidate_news_ctr = torch.squeeze(candidate_news_ctr)
            eta = torch.squeeze(torch.sigmoid(self.final_score_gate(user_final_embed)))
            score = (1-eta)*score_match+eta*candidate_news_ctr
        elif self.atten_op:
            user_target_embed, user_self_embed = user_embed
            b_size, l_size, e_size = user_self_embed.size(0), user_target_embed.size(1), user_self_embed.size(1)
            user_self_embed = torch.unsqueeze(user_self_embed, 1).expand(b_size, l_size, e_size)
            user_stack_embed = torch.stack([user_target_embed, user_self_embed], dim=2)
            user_final_embed = self.atten_op_addattr(user_stack_embed.view(-1, user_stack_embed.size(2), user_stack_embed.size(3)))
            user_final_embed = user_final_embed.view(-1, l_size, e_size)
            score = torch.matmul(news_embed, user_final_embed.permute([0, 2, 1])) # batch_size, negnums+1, negnums+1
            score_match = torch.squeeze(torch.diagonal(score, dim1=-2, dim2=-1)) # batch_size, negnums+1
            candidate_news_ctr = torch.squeeze(candidate_news_ctr)
            eta = torch.squeeze(torch.sigmoid(self.final_score_gate(user_final_embed)))
            score = (1-eta)*score_match+eta*candidate_news_ctr

        elif self.dnn:
            user_target_embed, user_self_embed = user_embed
            b_size, l_size, e_size = user_self_embed.size(0), user_target_embed.size(1), user_self_embed.size(1)
            user_self_embed = torch.unsqueeze(user_self_embed, 1).expand(b_size, l_size, e_size)
            user_concat_embed = torch.cat([user_target_embed, user_self_embed], dim=2)
            hidden_dnn = self.dnn_dense_1(user_concat_embed)
            user_final_embed = self.dnn_dense_2(hidden_dnn) 
            score = torch.matmul(news_embed, user_final_embed.permute([0, 2, 1])) # batch_size, negnums+1, negnums+1
            score_match = torch.squeeze(torch.diagonal(score, dim1=-2, dim2=-1)) # batch_size, negnums+1
            candidate_news_ctr = torch.squeeze(candidate_news_ctr)
            eta = torch.squeeze(torch.sigmoid(self.final_score_gate(user_final_embed)))
            score = (1-eta)*score_match+eta*candidate_news_ctr
        elif self.moe:
            user_target_embed, user_self_embed = user_embed
            b_size, l_size, e_size = user_self_embed.size(0), user_target_embed.size(1), user_self_embed.size(1)
            user_self_embed = torch.unsqueeze(user_self_embed, 1).expand(b_size, l_size, e_size)
            user_concat_embed = torch.cat([user_target_embed, user_self_embed], dim=2)
            user_outs = []
            for n in range(len(self.expert_nets)):
                user_hidden = self.expert_nets[n][0](user_concat_embed)
                user_outs.append(self.expert_nets[n][1](user_hidden))

            gate_hidden = self.gate_net_1(user_concat_embed)
            gate_out = self.gate_net_2(gate_hidden)
            gate_out = torch.softmax(gate_out, dim=-1)

            user_final_embed = torch.sum(torch.unsqueeze(gate_out, -1)*torch.stack(user_outs, dim=2), dim=2)
            if self.bias:
                hidden_dnn = self.bias_dense_1(user_concat_embed)
                user_final_embed_bias = self.bias_dense_2(hidden_dnn)
                score_main = torch.matmul(news_embed, user_final_embed.permute([0, 2, 1])) # batch_size, negnums+1, negnums+1
                score_bias = torch.matmul(news_embed, user_final_embed_bias.permute([0, 2, 1])) # batch_size, negnums+1, negnums+1
                score_main = torch.squeeze(torch.diagonal(score_main, dim1=-2, dim2=-1))
                score_bias = torch.squeeze(torch.diagonal(score_bias, dim1=-2, dim2=-1))
                candidate_news_ctr = torch.squeeze(candidate_news_ctr)
                eta = torch.squeeze(torch.sigmoid(self.final_score_gate(user_final_embed)))
                score =  (1-eta)*(score_main + score_bias) + eta*candidate_news_ctr
            else:
                score = torch.matmul(news_embed, user_final_embed.permute([0, 2, 1])) # batch_size, negnums+1, negnums+1
                score_match = torch.squeeze(torch.diagonal(score, dim1=-2, dim2=-1))
                candidate_news_ctr = torch.squeeze(candidate_news_ctr)
                eta = torch.squeeze(torch.sigmoid(self.final_score_gate(user_final_embed)))
                score = (1-eta)*score_match+eta*candidate_news_ctr
        elif self.mvke:
            user_target_embed, user_self_embed = user_embed
            b_size, l_size, e_size = user_self_embed.size(0), user_target_embed.size(1), user_self_embed.size(1)
            user_self_embed = torch.unsqueeze(user_self_embed, 1).expand(b_size, l_size, e_size)
            user_stack_embed = torch.stack([user_target_embed, user_self_embed], dim=2)
            user_stack_embed = user_stack_embed.view(-1, 2, e_size)
            user_outs = []
            for n in range(len(self.expert_nets)):
                user_q = self.user_qs[n,:]
                user_q = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(user_q, 0), 0), 0).expand(b_size, l_size, 1, e_size).view(-1, 1, e_size)
                _, atten_weights = self.qs_atten_nets[n](user_stack_embed, None, user_q, user_stack_embed, no_multi_head=True, return_w=True)
                atten_weights = torch.squeeze(atten_weights, 1)
                user_target_weighted_embed = torch.unsqueeze(atten_weights[:, 0], 1) * user_stack_embed[:, 0, :]
                user_self_weighted_embed = torch.unsqueeze(atten_weights[:, 1], 1) * user_stack_embed[:, 1, :]
                user_concat_weighted_embed = torch.cat([user_target_weighted_embed.view(b_size, -1, e_size), user_self_weighted_embed.view(b_size, -1, e_size)], dim=2)
                user_hidden = self.expert_nets[n][0](user_concat_weighted_embed)
                user_outs.append(self.expert_nets[n][1](user_hidden))

            vk_key = torch.unsqueeze(torch.unsqueeze(self.user_qs, 0), 0).expand(b_size, l_size, len(self.expert_nets), e_size).view(-1, len(self.expert_nets), e_size)
            exp_out = torch.stack(user_outs, dim=2).view(-1, len(self.expert_nets), e_size)
            user_final_embed = torch.squeeze(self.gate_net(vk_key, None, torch.unsqueeze(news_embed, 2).view(-1, 1, e_size), exp_out), 1).view(b_size, -1, e_size)
            score = torch.matmul(news_embed, user_final_embed.permute([0, 2, 1])) # batch_size, negnums+1, negnums+1
            score_match = torch.squeeze(torch.diagonal(score, dim1=-2, dim2=-1))
            candidate_news_ctr = torch.squeeze(candidate_news_ctr)
            eta = torch.squeeze(torch.sigmoid(self.final_score_gate(user_final_embed)))
            score = (1-eta)*score_match+eta*candidate_news_ctr
        elif self.cross_gate:
            user_target_embed, user_self_embed = user_embed
            b_size, l_size, e_size = user_self_embed.size(0), user_target_embed.size(1), user_self_embed.size(1)
            user_self_embed = torch.unsqueeze(user_self_embed, 1).expand(b_size, l_size, e_size)
            user_concat_embed = torch.cat([user_target_embed, user_self_embed], dim=2)
            cross_gate_param = torch.sigmoid(self.cross_gate_dense(user_concat_embed))
            user_final_embed = cross_gate_param * user_self_embed
            score = torch.matmul(news_embed, user_final_embed.permute([0, 2, 1])) # batch_size, negnums+1, negnums+1
            score_match = torch.squeeze(torch.diagonal(score, dim1=-2, dim2=-1)) # batch_size, negnums+1
            candidate_news_ctr = torch.squeeze(candidate_news_ctr)
            eta = torch.squeeze(torch.sigmoid(self.final_score_gate(user_final_embed)))
            score = (1-eta)*score_match+eta*candidate_news_ctr
        else:
            if len(user_embed.shape) == 3: # batch_size, negnums+1, num_heads*head_size+2*categ_embed_size
                score_raw = torch.matmul(news_embed, user_embed.permute([0, 2, 1])) # batch_size, negnums+1, negnums+1
                if self.score_gate:
                    score_match = torch.squeeze(torch.diagonal(score_raw, dim1=-2, dim2=-1)) # batch_size, negnums+1
                    candidate_news_ctr = torch.squeeze(candidate_news_ctr)
                    eta = torch.squeeze(torch.sigmoid(self.user_ctr_dense(user_embed)))
                    score = (1-eta)*score_match+eta*candidate_news_ctr
                else:
                    score = torch.squeeze(torch.diagonal(score_raw, dim1=-2, dim2=-1)) # batch_size, negnums+1
            else:
                user_embed = torch.unsqueeze(user_embed, 2) # batch_size, num_heads*head_size+2*categ_embed_size, 1
                if self.score_gate:
                    score_match = torch.squeeze(torch.matmul(news_embed, user_embed)) # batch_size, negnums+1
                    candidate_news_ctr = torch.squeeze(candidate_news_ctr)
                    eta = torch.squeeze(torch.sigmoid(self.user_ctr_dense(torch.squeeze(user_embed, 2))), 0)
                    score = (1-eta)*score_match+eta*candidate_news_ctr
                else:
                    score = torch.squeeze(torch.matmul(news_embed, user_embed)) # batch_size, negnums+1
        

        if labels is not None:
            loss = self.criterion(score, labels)
            if self.util_reg:
                loss += user_embed[1]
            return loss
        else:
            score = torch.sigmoid(score) # num_impression,
            return score

class NRMS(nn.Module):
    def __init__(self, preembed, args, logger, data):
        super(NRMS, self).__init__()
        args.device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
        if args.dataset == 'heybox':
            userinfo = len(data.post_user_dict)
        else:
            userinfo = None
        log_user_info = len(data.users)
        num_ctr_bin = 11
        self.model = nrms(data.word_dict, preembed, len(data.categ_dict), userinfo, log_user_info, num_ctr_bin, args).to(args.device)
        self.args = args
        self.logger = logger
        self.data = data

    def mtrain(self):
        args = self.args
        batch_num = len(self.data.train_label)//args.batch_size
        args.max_steps = args.epochs * batch_num

        if (args.optimizer == 'Adamw'):
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2)
        elif (args.optimizer == 'Adam'):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.l2)
        elif (args.optimizer == 'SGD'):
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.l2)

        args.n_gpu = torch.cuda.device_count()
        if args.use_multi_gpu and args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        global_step = 0
        best_auc = 0
        self.model.train()
        for epoch in range(args.epochs):
            train_loss = 0
            
            start_train = time.time()
            train_progress = tqdm(enumerate(self.data.generate_batch_train_data()), dynamic_ncols=True,
                                  total=batch_num)
            for step, batch in train_progress:
                news, news_ctr, user_click, click_len, use_id, labels = (torch.LongTensor(x).to(args.device) for x in batch)
                del batch
                optimizer.zero_grad()
                loss = self.model(news, news_ctr, user_click, click_len, use_id, labels, 0)
                if args.use_multi_gpu and args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                if args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), args.max_grad_norm)
                train_loss += loss.item()
                optimizer.step()
                global_step += 1
                train_progress.set_description(u"[{}] Loss: {:,.6f} ----- ".format(epoch, train_loss / (step + 1)))

            self.logger.info('Time taken for training 1 epoch {} sec'.format(time.time() - start_train))
            self.logger.info('epoch:{}, loss:{}'.format(epoch, train_loss / batch_num))

            start_eval = time.time()
            preds = self.infer() # num_test_smaples, num_impression
            auc, mrr, ndcg5, ndcg10 = self.getscore(preds, self.data.eval_label)
            self.logger.info('Time taken for testing 1 epoch {} sec'.format(time.time() - start_eval))
            self.logger.info('auc:{}, mrr:{}, ndcg5:{}, ndcg10:{}'.format(auc, mrr, ndcg5, ndcg10))

            if auc > best_auc:
                # test and save
                if args.save == 1:
                    model_to_save = self.model.module if hasattr(self.model,
                                                                 'module') else self.model  # Only save the model it-self
                    output_model_file = os.path.join(args.savepath, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                best_auc = auc
                best_mrr = mrr
                best_ndcg5 = ndcg5
                best_ndcg10 = ndcg10
                self.logger.info('Best performance: auc:{}, mrr:{}, ndcg5:{}, ndcg10:{}'.format(best_auc, best_mrr, best_ndcg5, best_ndcg10))

    def infer(self):
        args = self.args
        self.model.eval()
        predict = []
        eval_progress = tqdm(enumerate(self.data.generate_batch_eval_data()), dynamic_ncols=True,
                             total=(len(self.data.eval_label) // args.eval_batch_size))
        for step, batch in eval_progress:
            news, news_ctr, user_click, click_len, use_id = (torch.LongTensor(x).to(args.device) for x in batch)
            with torch.no_grad():
                click_probability = self.model(news, news_ctr, user_click, click_len, use_id, train_test_flag=1)

            predict.append(click_probability.cpu().numpy())

        return predict # num_test_smaples, num_impression

    def getscore(self, preds, labels): # num_test_smaples, num_impression; num_test_smaples, num_impression
        aucs, mrrs, ndcg5s, ndcg10s = 0, 0, 0, 0
        testnum = len(labels)
        for i in range(testnum):
            aucs += roc_auc_score(labels[i], preds[i])
            mrrs += mrr_score(labels[i], preds[i])
            ndcg5s += ndcg_score(labels[i], preds[i], 5)
            ndcg10s += ndcg_score(labels[i], preds[i], 10)
        return aucs / testnum, mrrs / testnum, ndcg5s / testnum, ndcg10s / testnum
