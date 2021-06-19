import random
import json
import transformers as _
from transformers1 import BertTokenizer
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from itertools import chain

def writeToJsonFile(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False,indent=0))
def readFromJsonFile(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())

def loadData(path):
    allData=[]
    with open(path,"r") as f:
        for i in f:
            i=i.strip().split('\t')
            if len(i)==0:#防止空行
                break
            if len(i)==3:#训练集
                a,b,label=i
                a=a.split(' ')
                b=b.split(' ')
            else:#测试集，直接转为id形式
                a,b,label=i[0],i[1],-1
                a=a.split(' ')
                b=b.split(' ')
            allData.append([a,b,label])
    return allData

def calNegPos(ls):#计算正负比例
    posNum,negNum=0,0
    for i in ls:
        if i[2]==0:
            negNum+=1
        elif i[2]==1:
            posNum+=1
    posNum=1 if posNum==0 else posNum
    return negNum,posNum,round(negNum/posNum,4)

allData=loadData('/tcdata/gaiic_track3_round1_train_20210228.tsv')+loadData('/tcdata/gaiic_track3_round2_train_20210407.tsv')
testA_data = loadData('/tcdata/gaiic_track3_round1_testA_20210228.tsv')
testB_data = loadData('/tcdata/gaiic_track3_round1_testB_20210317.tsv')
random.shuffle(allData)

train_data=allData+testA_data+testB_data#全量
valid_data=allData[-20000:]
print("训练集样本数量：", len(train_data))

def paddingList(ls:list,val,returnTensor=False):
    ls=ls[:]#不要改变了原list尺寸
    maxLen=max([len(i) for i in ls])
    for i in range(len(ls)):
        ls[i]=ls[i]+[val]*(maxLen-len(ls[i]))
    return torch.tensor(ls,device='cuda') if returnTensor else ls

def truncate(a:list,b:list,maxLen):
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
    return a,b

class MLM_Data(Dataset):
    #传入句子对列表
    def __init__(self,textLs:list,maxLen:int,tk:BertTokenizer):
        super().__init__()
        self.data=textLs
        self.maxLen=maxLen
        self.tk=tk
        self.spNum=len(tk.all_special_tokens)
        self.tkNum=tk.vocab_size

    def __len__(self):
        return len(self.data)

    def random_mask(self,text_ids):
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        idx=0
        while idx<len(rands):
            if rands[idx]<0.15:#需要mask
                ngram=np.random.choice([1,2,3], p=[0.7,0.2,0.1])#若要mask，进行x_gram mask的概率
                if ngram==3 and len(rands)<7:#太大的gram不要应用于过短文本
                    ngram=2
                if ngram==2 and len(rands)<4:
                    ngram=1
                L=idx+1
                R=idx+ngram#最终需要mask的右边界（开）
                while L<R and L<len(rands):
                    rands[L]=np.random.random()*0.15#强制mask
                    L+=1
                idx=R
                if idx<len(rands):
                    rands[idx]=1#禁止mask片段的下一个token被mask，防止一大片连续mask
            idx+=1

        for r, i in zip(rands, text_ids):
            if r < 0.15 * 0.8:
                input_ids.append(self.tk.mask_token_id)
                output_ids.append(i)#mask预测自己
            elif r < 0.15 * 0.9:
                input_ids.append(i)
                output_ids.append(i)#自己预测自己
            elif r < 0.15:
                input_ids.append(np.random.randint(self.spNum,self.tkNum))
                output_ids.append(i)#随机的一个词预测自己，随机词不会从特殊符号中选取，有小概率抽到自己
            else:
                input_ids.append(i)
                output_ids.append(-100)#保持原样不预测

        return input_ids, output_ids

    #耗时操作在此进行，可用上多进程
    def __getitem__(self, item):
        text1,text2,_=self.data[item]#预处理，mask等操作
        if random.random()>0.5:
            text1,text2=text2,text1#交换位置
        text1,text2=truncate(text1,text2,self.maxLen)
        text1_ids,text2_ids = self.tk.convert_tokens_to_ids(text1),self.tk.convert_tokens_to_ids(text2)
        text1_ids, out1_ids = self.random_mask(text1_ids)#添加mask预测
        text2_ids, out2_ids = self.random_mask(text2_ids)
        input_ids = [self.tk.cls_token_id] + text1_ids + [self.tk.sep_token_id] + text2_ids + [self.tk.sep_token_id]#拼接
        token_type_ids=[0]*(len(text1_ids)+2)+[1]*(len(text2_ids)+1)
        labels = [-100] + out1_ids + [-100] + out2_ids + [-100]
        assert len(input_ids)==len(token_type_ids)==len(labels)
        return {'input_ids':input_ids,'token_type_ids':token_type_ids,'labels':labels}

    @classmethod
    def collate(cls,batch):
        input_ids=[i['input_ids'] for i in batch]
        token_type_ids=[i['token_type_ids'] for i in batch]
        labels=[i['labels'] for i in batch]
        input_ids=paddingList(input_ids,0,returnTensor=True)
        token_type_ids=paddingList(token_type_ids,0,returnTensor=True)
        labels=paddingList(labels,-100,returnTensor=True)
        attention_mask=(input_ids!=0)
        return {'input_ids':input_ids,'token_type_ids':token_type_ids
                ,'attention_mask':attention_mask,'labels':labels}




unionList=lambda ls:list(chain(*ls))#按元素拼接
splitList=lambda x,bs:[x[i:i+bs] for i in range(0,len(x),bs)]#按bs切分


#sortBsNum：原序列按多少个bs块为单位排序，可用来增强随机性
#比如如果每次打乱后都全体一起排序，那每次都是一样的
def blockShuffle(data:list,bs:int,sortBsNum,key):
    random.shuffle(data)#先打乱
    tail=len(data)%bs#计算碎片长度
    tail=[] if tail==0 else data[-tail:]
    data=data[:len(data)-len(tail)]
    assert len(data)%bs==0#剩下的一定能被bs整除
    sortBsNum=len(data)//bs if sortBsNum is None else sortBsNum#为None就是整体排序
    data=splitList(data,sortBsNum*bs)
    data=[sorted(i,key=key,reverse=True) for i in data]#每个大块进行降排序
    data=unionList(data)
    data=splitList(data,bs)#最后，按bs分块
    random.shuffle(data)#块间打乱
    data=unionList(data)+tail
    return data
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter,_MultiProcessingDataLoaderIter
#每轮迭代重新分块shuffle数据的DataLoader
class blockShuffleDataLoader(DataLoader):
    def __init__(self, dataset: Dataset,sortBsNum,key,**kwargs):
        assert isinstance(dataset.data,list)#需要有list类型的data属性
        super().__init__(dataset,**kwargs)#父类的参数传过去
        self.sortBsNum=sortBsNum
        self.key=key

    def __iter__(self):
        #分块shuffle
        self.dataset.data=blockShuffle(self.dataset.data,self.batch_size,self.sortBsNum,self.key)
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)
