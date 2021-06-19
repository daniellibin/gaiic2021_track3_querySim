import torch
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer, XLNetConfig
import numpy as np
import os
import random
from Config import *
import torch
import torch.nn as nn
import torch.nn.functional as F

def paddingList(ls:list,val,returnTensor=False):
    ls=ls[:]#不要改变了原list尺寸
    maxLen=max([len(i) for i in ls])
    for i in range(len(ls)):
        ls[i]=ls[i]+[val]*(maxLen-len(ls[i]))
    return torch.tensor(ls,device='cuda') if returnTensor else ls

def fastTokenizer(a:str,b:str,maxLen,tk):
    a,b=a.split(),b.split()
    a,b=tk.convert_tokens_to_ids(a),tk.convert_tokens_to_ids(b)
    maxLen-=3#空留给cls sep sep
    assert maxLen>=0
    len2=maxLen//2#若为奇数，更长部分给左边
    len1=maxLen-len2
    #一共就a超长与否，b超长与否，组合的四种情况
    if len(a)+len(b)>maxLen:#需要截断
        if len(a)<=len1 and len(b)>len2:
            b=b[:maxLen-len(a)]
        elif len(a)>len1 and len(b)<=len2:
            a=a[:maxLen-len(b)]
        elif len(a)>len1 and len(b)>len2:
            a=a[:len1]
            b=b[:len2]
    input_ids=[tk.cls_token_id]+a+[tk.sep_token_id]+b+[tk.sep_token_id]
    token_type_ids=[0]*(len(a)+2)+[1]*(len(b)+1)
    return {'input_ids': input_ids, 'token_type_ids': token_type_ids}

class data_generator:
    def __init__(self, data, config, shuffle=False):
        self.data = data
        self.batch_size = config.batch_size
        self.max_length = config.MAX_LEN
        self.shuffle = shuffle

        vocab = 'vocab.txt' if os.path.exists(config.model_path + 'vocab.txt') else 'spiece.model'
        self.tokenizer = TOKENIZERS[config.model].from_pretrained(config.model_path + vocab)

        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        q1, q2, y = self.data
        idxs = list(range(len(self.data[0])))
        if self.shuffle:
            np.random.shuffle(idxs)
        input_ids, input_masks, segment_ids, labels = [], [], [], []

        for index, i in enumerate(idxs):

            text = q1[i]
            text_pair = q2[i]
            '''
            # text = self.tokenizer(text, text_pair, padding='max_length', truncation=True, max_length=self.max_length)
            text = fastTokenizer(text, text_pair, self.max_length, self.tokenizer)
            input_ids.append(text['input_ids'])
            segment_ids.append(text['token_type_ids'])
            input_masks.append([1] * len(text['input_ids']))  # bs为1时无padding，全1
            yield input_ids, input_masks, segment_ids, labels
            input_ids, input_masks, segment_ids, labels = [], [], [], []

            '''
            tkRes = self.tokenizer(text, text_pair, max_length=self.max_length, truncation='longest_first',
                                   return_attention_mask=False)
            input_id = tkRes['input_ids']
            segment_id = tkRes['token_type_ids']
            assert len(segment_id) == len(input_id)
            input_ids.append(input_id)
            segment_ids.append(segment_id)
            labels.append(y[i])

            if len(input_ids) == self.batch_size or i == idxs[-1]:
                input_ids = paddingList(input_ids, 0, returnTensor=True)  # 动态padding
                segment_ids = paddingList(segment_ids, 0, returnTensor=True)
                input_masks = (input_ids != 0)
                yield input_ids, input_masks, segment_ids, labels
                input_ids, input_masks, segment_ids, labels = [], [], [], []



class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=0.3, alpha=0.1, emb_name='word_embeddings', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]



class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.25, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# 支持多分类和二分类
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss
    for well-classified examples (p>0.5) putting more
    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index,
    should be specific when alpha is float
    :param size_average: (bool, optional) By default,
    the losses are averaged over each loss element in the batch.
    """
    def __init__(self, num_class, alpha=None, gamma=2,
                smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        else:
            raise TypeError('Not support alpha type')
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def f1_match(y_true,y_pred):
    acc = sum(y_pred & y_true) / (sum(y_pred))
    rec = sum(y_pred & y_true) / (sum(y_true))

    return 2 * acc * rec /(acc + rec)