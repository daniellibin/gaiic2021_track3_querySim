# coding:utf-8
import numpy as np
import random
import os
random.seed(0)
np.random.seed(0)#seed应该在main里尽早设置，以防万一
os.environ['PYTHONHASHSEED'] =str(0)#消除hash算法的随机性
import transformers as _
from transformers1 import Trainer, TrainingArguments,BertTokenizer
from NLP_Utils import MLM_Data,train_data,blockShuffleDataLoader

from NEZHA.configuration_nezha import NeZhaConfig
from NEZHA.modeling_nezha import NeZhaForMaskedLM

maxlen=100
batch_size=128
vocab_file_dir = './nezha_model/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)

config = NeZhaConfig(
    vocab_size=len(tokenizer),
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512,
)



model = NeZhaForMaskedLM.from_pretrained("../../nezha-cn-base/")

model.resize_token_embeddings(len(tokenizer))
print(model)
train_MLM_data=MLM_Data(train_data,maxlen,tokenizer)
#自己定义dataloader，不要用huggingface的
dl=blockShuffleDataLoader(train_MLM_data,None,key=lambda x:len(x[0])+len(x[1]),shuffle=False
                          ,batch_size=batch_size,collate_fn=train_MLM_data.collate)

training_args = TrainingArguments(
    output_dir='./nezha_output',
    overwrite_output_dir=True,
    num_train_epochs=400,
    per_device_train_batch_size=batch_size,
    save_steps=len(dl)*10000,#每10个epoch save一次
    save_total_limit=3,
    logging_steps=len(dl),#每个epoch log一次
    seed=2021,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=int(450000*150/batch_size*0.03)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataLoader=dl,
    prediction_loss_only=True,
)

if __name__ == '__main__':
    trainer.train()
    trainer.save_model('./nezha_model')
