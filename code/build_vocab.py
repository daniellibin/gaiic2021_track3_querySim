from collections import Counter
def loadData(path):
    allData=[]
    with open(path,"r") as f:
        for i in f:
            i=i.strip().split('\t')
            if len(i)==0:#防止空行
                break
            if len(i)==3:#训练集
                a,b,label=i
            else:#测试集，直接转为id形式
                a,b,label=i[0],i[1],-1
            a,b=[int(i) for i in a.split()],[int(i) for i in b.split()]
            allData.append([a,b])
    return allData

allData=loadData('/tcdata/gaiic_track3_round1_train_20210228.tsv')+loadData('/tcdata/gaiic_track3_round2_train_20210407.tsv')
test_data = loadData('/tcdata/gaiic_track3_round1_testA_20210228.tsv')+loadData('/tcdata/gaiic_track3_round1_testB_20210317.tsv')

model_lists = ["nezha-base-count3", "nezha-base-count5", "bert-base-count3",
               "bert-base-count3-len100", "bert-base-count5", "bert-base-count5-len32"]
childPath_lists=[
    ['/pretrain/nezha_model/','/finetuning/models/'],
    ['/pretrain/nezha_model/','/finetuning/models/'],
    ['/pretrain/bert_model/','/finetuning/models/'],

    ['/finetuning/models/'],
    ['/pretrain/bert_model/','/finetuning/models/'],
    ['/finetuning/models/'],
           ]
counts=[3,5,3,3,5,5]

token2count=Counter()
for i,j in allData+test_data:
    token2count.update(i+j)

for modelPath,childPath,ct in zip(model_lists,childPath_lists,counts):
    pre=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]',]
    tail=[]
    for k,v in token2count.items():
        if v>=ct:
            tail.append(k)
    tail.sort()
    vocab=pre+tail
    print(f"模型{modelPath}，词频：{ct}，词表大小：{len(vocab)}")
    for ch in childPath:
        with open(modelPath+ch+'vocab.txt', "w", encoding="utf-8") as f:
            for i in vocab:
                f.write(str(i)+'\n')

















