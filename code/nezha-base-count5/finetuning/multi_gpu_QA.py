from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import logging
import torch
import random
import os
from torch import nn, optim
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer, XLNetConfig
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, roc_auc_score
from model import *
from utils import *
import time

import logging
logging.basicConfig(level=logging.DEBUG, filename="train.log",filemode='a')


from NEZHA.modeling_nezha import *

MODEL_CLASSES = {
    'BertForClass': BertForClass,
    'BertLastCls': BertLastCls,
    'BertLastTwoCls': BertLastTwoCls,
    'BertLastTwoClsPooler': BertLastTwoClsPooler,
    'BertLastTwoEmbeddings': BertLastTwoEmbeddings,
    'BertLastTwoEmbeddingsPooler': BertLastTwoEmbeddingsPooler,
    'BertLastFourCls': BertLastFourCls,
    'BertLastFourClsPooler': BertLastFourClsPooler,
    'BertLastFourEmbeddings': BertLastFourEmbeddings,
    'BertLastFourEmbeddingsPooler': BertLastFourEmbeddingsPooler,
    'BertDynCls': BertDynCls,
    'BertDynEmbeddings': BertDynEmbeddings,
    'BertRNN': BertRNN,
    'BertCNN': BertCNN,
    'BertRCNN': BertRCNN,
    'XLNet': XLNet,
    'Electra': Electra,
    'NEZHA': NEZHA,

}


class Config:
    def __init__(self):
        # 预训练模型路径
        self.modelId = 2
        self.model = "NEZHA"
        self.Stratification = False
        self.model_path = '../pretrain/nezha_model/'

        self.num_class = 2
        self.dropout = 0.2
        self.MAX_LEN = 100
        self.epoch = 3
        self.learn_rate = 4e-5
        self.normal_lr = 1e-4
        self.batch_size = 32
        self.k_fold = 10
        self.seed = 42

        self.device = torch.device('cuda')
        # self.device = torch.device('cpu')

        self.focalloss = False
        self.pgd = False
        self.fgm = True


config = Config()
os.environ['PYTHONHASHSEED']='0'#消除hash算法的随机性
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)


file_path = './log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)


train = pd.read_csv('/tcdata/gaiic_track3_round1_train_20210228.tsv',sep='\t',header=None)
semi = pd.read_csv('/tcdata/gaiic_track3_round2_train_20210407.tsv',sep='\t',header=None)
train = pd.concat([train, semi], sort=False)
train.columns=['q1','q2','label']


train_query1 = train['q1'].values.astype(str)
train_query2 = train['q2'].values.astype(str)
train_label = train['label'].values.astype(int)


oof_train = np.zeros((len(train), config.num_class), dtype=np.float32)


#kf = StratifiedKFold(n_splits=config.k_fold, shuffle=True, random_state=config.seed)
kf = KFold(n_splits=config.k_fold, shuffle=True, random_state=config.seed)

for fold, (train_index, valid_index) in enumerate(kf.split(train_query1, train_label)):

    print('\n\n------------fold:{}------------\n'.format(fold))

    '''
    q1 = train_query1[train_index]
    q2 = train_query2[train_index]
    y = train_label[train_index]
    '''
    q1 = train_query1
    q2 = train_query2
    y = train_label


    val_q1 = train_query1[valid_index]
    val_q2 = train_query2[valid_index]
    val_y = train_label[valid_index]

    train_D = data_generator([q1, q2, y], config, shuffle=True)
    val_D = data_generator([val_q1, val_q2, val_y], config)

    model = MODEL_CLASSES[config.model](config).to(config.device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)


    if config.pgd:
        pgd = PGD(model)
        K = 3

    elif config.fgm:
        fgm = FGM(model)

    if config.focalloss:
        loss_fn = FocalLoss(config.num_class)
    else:
        loss_fn = nn.CrossEntropyLoss()  # BCEWithLogitsLoss就是把Sigmoid-BCELoss合成一步


    num_train_steps = int(len(train) / config.batch_size * config.epoch)
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if config.Stratification:
        bert_params = [x for x in param_optimizer if 'bert' in x[0]]
        normal_params = [p for n, p in param_optimizer if 'bert' not in n]
        optimizer_parameters = [
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': normal_params, 'lr': config.normal_lr},
        ]
    else:
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

    optimizer = AdamW(optimizer_parameters, lr=config.learn_rate) # lr为全局学习率
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train) / config.batch_size / 2),
        num_training_steps=num_train_steps
    )

    best_auc = 0
    PATH = './models/bert_{}.pth'.format(fold)
    save_model_path = './models/'
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    for e in range(config.epoch):
        print('\n------------epoch:{}------------'.format(e))
        model.train()
        acc = 0
        train_len = 0
        loss_num = 0
        tq = tqdm(train_D,ncols=70,disable=True)
        last=time.time()
        for input_ids, input_masks, segment_ids, labels in tq:
            label_t = torch.tensor(labels, dtype=torch.long).to(config.device)

            y_pred = model(input_ids, input_masks, segment_ids)

            loss = loss_fn(y_pred, label_t)
            loss = loss.mean()
            loss.backward()

            if config.pgd:
                pgd.backup_grad()
                # 对抗训练
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    y_pred = model(input_ids, input_masks, segment_ids)

                    loss_adv = loss_fn(y_pred, label_t)
                    loss_adv = loss_adv.mean()
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore()  # 恢复embedding参数

            elif config.fgm:
                # 对抗训练
                fgm.attack()  # 在embedding上添加对抗扰动
                y_pred = model(input_ids, input_masks, segment_ids)
                loss_adv = loss_fn(y_pred, label_t)
                loss_adv = loss_adv.mean()
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复embedding参数


            # 梯度下降，更新参数
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            y_pred = np.argmax(y_pred.detach().to("cpu").numpy(), axis=1)
            acc += sum(y_pred == labels)
            loss_num += loss.item()
            train_len += len(labels)
            tq.set_postfix(fold=fold, epoch=e, loss=loss_num / train_len, acc=acc / train_len)
        print(f"微调第{e}轮耗时：{time.time()-last}")
        model.eval()
        with torch.no_grad():
            y_p = []
            y_l = []
            train_logit = None
            for input_ids, input_masks, segment_ids, labels in tqdm(val_D,disable=True):
                label_t = torch.tensor(labels, dtype=torch.long).to(config.device)

                y_pred = model(input_ids, input_masks, segment_ids)
                y_pred = F.softmax(y_pred)
                y_pred = y_pred.detach().to("cpu").numpy()
                if train_logit is None:
                    train_logit = y_pred
                else:
                    train_logit = np.vstack((train_logit, y_pred))

                y_p += list(y_pred[:,1])

                y_pred = np.argmax(y_pred, axis=1)
                y_l += list(y_pred)


            f1 = f1_score(val_y, y_l, average="macro")
            auc_score = roc_auc_score(val_y, y_p)
            print("best_auc:{}  auc_score:{}  f1:{}\n".format(best_auc, auc_score, f1))
            if auc_score >= best_auc:
                best_auc = auc_score
                oof_train[valid_index] = np.array(train_logit)
                #torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(), PATH)
                torch.save(model.module if hasattr(model, "module") else model, PATH)

    optimizer.zero_grad()

    del model
    torch.cuda.empty_cache()

    break

