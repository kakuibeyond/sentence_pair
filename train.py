__author__='zhangbei'
__date__='2020/10/29'

import os
import csv
import tqdm
import time
import random
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F
import tokenizers
import transformers
from transformers import BertConfig, BertForSequenceClassification
from model_bert import RobertaModel


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(8)

filemap = {
    'train': './data/process_data/train.train_raw.tsv',
    'dev': './data/process_data/train.dev.tsv',
    'test': './data/process_data/test.tsv'
}

# 读取数据，返回query文本，reply文本，label标签(0/1)
def read_data(mode='train'):
    filename = filemap[mode]
    queries, replies,labels=[],[],[]
    texts=[]
    with open(filename,encoding='utf-8') as f:
        f_csv=csv.reader(f, delimiter='\t')
        for line in f_csv:
            texts.append(line)
    for text in tqdm.tqdm(texts,desc='loading '+filename):
        queries.append(text[2])
        replies.append(text[3])
        if mode=='test':
            labels.append(0)
        else:
            labels.append(int(text[4]))
    return queries,replies,labels


# 封装数据集类，返回的token.ids,token.label,token.attention_mask组成的tuple
class Dataset(torch.utils.data.Dataset):
    def __init__(self,mode):
        self.tokenizer=tokenizers.BertWordPieceTokenizer("./model/RoBERTa/vocab.txt")
        self.tokenizer.enable_padding(length=64)
        self.tokenizer.enable_truncation(max_length=64)

        self.queries, self.replies, self.labels=read_data(mode=mode)

    def __getitem__(self,index):
        token=self.tokenizer.encode(self.queries[index], self.replies[index])
        token.label=self.labels[index]

        return [torch.tensor(i) for i in (token.ids,token.label,token.attention_mask)]

    def __len__(self):
        return len(self.queries)


class CLASSIFIER:
    def __init__(self, config, load_pretrained=True):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RobertaModel(config).to(self.device)# 模型结构确定
        # self.model = BertForSequenceClassification(config).to(self.device)
        if load_pretrained:
            # 载入预训练权重
            roberta_state=torch.load('model/RoBERTa/chinese-roberta-wwm-ext.bin', map_location=self.device).state_dict()
            self.model.roberta.load_state_dict(roberta_state)

    #在训练过程中实时验证模型的好坏
    def val(self,loader_test):
        self.model.eval()
        y = np.array([])
        pred = np.array([])
        loss = 0
        with torch.no_grad():
            for step, (batch_x,batch_y,batch_mask) in enumerate(loader_test):
                y = np.append(y, batch_y)
                batch_max_length=batch_mask.sum(1).max().item()
                batch_x=batch_x[:,:batch_max_length].to(self.device)
                batch_mask=batch_mask[:,:batch_max_length].to(self.device)

                batch_output = self.model(batch_x,batch_mask.byte()).cpu().flatten()
                loss += F.binary_cross_entropy_with_logits(batch_output, batch_y.type_as(batch_output)).item()
                batch_probs = batch_output.sigmoid().numpy()
                batch_pred = [1 if i>0.5 else 0 for i in batch_probs]
                pred = np.append(pred, batch_pred)
        loss /= step+1# 返回的loss为每个batch的平均
        self.model.train()#验证之后需要将模型继续变为train模式，后面继续训练
        precision = precision_score(y_true=y, y_pred=pred)
        recall = recall_score(y_true=y, y_pred=pred)
        f1 = f1_score(y_true=y, y_pred=pred)
        return precision,recall,f1,loss

    # 根据训练好的本地模型预测真测试集，测试集没有真标签，因此这里返回的precision, recall, f1无意义，仅为了得到pred
    def predict(self, check_point, mode):
        loader_test=torch.utils.data.DataLoader(dataset=Dataset(mode),batch_size=128,shuffle=False,num_workers=0)
        pth=torch.load(check_point,map_location='cpu')
        self.model.load_state_dict(pth['weights'])
        self.model.eval()
        y = np.array([])
        pred = np.array([])
        probs = np.array([])
        with torch.no_grad():
            for step, (batch_x,batch_y,batch_mask) in enumerate(loader_test):
                y = np.append(y, batch_y.numpy())
                batch_max_length=batch_mask.sum(1).max().item()
                batch_x=batch_x[:,:batch_max_length].to(self.device)
                batch_mask=batch_mask[:,:batch_max_length].to(self.device)

                batch_output = self.model(batch_x,batch_mask.byte()).cpu().flatten()
                batch_probs = batch_output.sigmoid().numpy()
                batch_pred = [1 if i>0.5 else 0 for i in batch_probs]
                pred = np.append(pred, batch_pred)
                probs = np.append(probs, batch_probs)
        precision = precision_score(y_true=y, y_pred=pred)
        recall = recall_score(y_true=y, y_pred=pred)
        f1 = f1_score(y_true=y, y_pred=pred)

        print('precision: {}\nrecall: {}\nf1 score: {}'.format(precision, recall, f1))
        return pred, probs, precision, recall, f1

    # 这里也有check_point，可以在未完成的基础上继续训练
    def train(self,check_point='',epochs=10,batch_size=16):
        loader_train=torch.utils.data.DataLoader(dataset=Dataset('train'),batch_size=batch_size,shuffle=True,num_workers=0)
        loader_test=torch.utils.data.DataLoader(dataset=Dataset('dev'),batch_size=128,shuffle=False,num_workers=0)

        optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-5)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=1e-8)
        
        if check_point !='':
            pth=torch.load(check_point,map_location='cpu')
            self.model.load_state_dict(pth['weights'])
            optimizer.load_state_dict(pth['optimizer'])

        model_save_names=['','','']
        f1_scores=[0,0,0]
        self.model.train()
        for epoch in range(epochs):
            loader_t=tqdm.tqdm(loader_train,desc='epoch:{}/{}'.format(epoch,epochs))
            for step,(batch_x,batch_y,batch_mask) in enumerate(loader_t):
                batch_max_length=batch_mask.sum(1).max().item()
                batch_x=batch_x[:,:batch_max_length].to(self.device)
                batch_y=batch_y.to(self.device)
                batch_mask=batch_mask[:,:batch_max_length].to(self.device)
                
                batch_pred=self.model(batch_x,batch_mask.byte()).flatten()
                loss = F.binary_cross_entropy_with_logits(batch_pred, batch_y.type_as(batch_pred))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loader_t.set_postfix(training="loss:{:.6f}".format(loss.item()))

                # 每500个batch尝试评估模型，并实时保存最好的3个模型
                if step%500==0:
                    pth={'weights':self.model.state_dict(),'optimizer':optimizer.state_dict()}

                    # if torch.os.path.exists(model_save_names[0]): torch.os.remove(model_save_names[0])
                    # model_save_names.append('epoch{}_batch{}.pth'.format(epoch, step))
                    # torch.save(pth,model_save_names[-1])
                    # model_save_names=model_save_names[1:]

                    precision,recall,f1, loss=self.val(loader_test)
                    print('precision: {:.4f}\trecall: {:.4f}\tf1_score: {:.4f}\tloss: {:.4f}'.format(precision, recall, f1, loss))

                    if f1>0.6 and f1>f1_scores[-1]:
                        f1_scores.append(f1)
                        # if torch.os.path.exists('f1_score_{:.4f}.pth'.format(f1_scores[0])): torch.os.remove('f1_score_{:.4f}.pth'.format(f1_scores[0]))
                        torch.save(pth,'roberta_cls_checkpoint/epoch{}_batch{}_f1_score_{:.4f}.pth'.format(epoch, step, f1_scores[-1]))
                        f1_scores=f1_scores[1:]

            scheduler.step()


# 根据训练好并已保存的模型进行预测分析真正的测试集，并生成提交文件
def predict_analysis(check_point, mode, submission_file):
    config_roberta = BertConfig.from_pretrained('model/RoBERTa/config.json')
    classifier=CLASSIFIER(config=config_roberta, load_pretrained=True)
    pred, probs, precision, recall, f1 = classifier.predict(check_point=check_point, mode=mode)
    texts=[]
    filename = filemap[mode]
    if mode=='test':
        df_test = pd.read_csv(filename,sep='\t',header=None)
        df_test.columns = ['qid','rid','q','r']
        df_test['label'] = pred.astype(int)
        df_test[['qid','rid','label']].to_csv(submission_file,index=False, header=None, sep='\t')

    # with open(filename,encoding='utf-8') as f:
    #     f_csv=csv.reader(f, delimiter='\t')
    #     for line in f_csv:
    #         texts.append(line)
    # with open(result_file, 'w', encoding='utf-8') as writefile:
    #     for index, line in enumerate(texts):
    #         true_label = 0 if mode=='test' else line[4]
    #         writefile.write('{}\t{}\t{}\t{}\t{}\n'.format(int(pred[index]), probs[index], true_label, line[2], line[3]))
    # with open(scorefile, 'w', encoding='utf-8') as f:
    #     f.write('precision: {}\nrecall: {}\nf1 score: {}'.format(precision, recall, f1))


if __name__ == "__main__":
    cpt_path = 'roberta_cls_checkpoint'
    sub_path = 'submission'
    if not os.path.exists(sub_path):
        os.mkdir(sub_path)
    if not os.path.exists(cpt_path):
        os.mkdir(cpt_path)
    # config_roberta = BertConfig.from_pretrained('model/RoBERTa/config.json')
    # classifier=CLASSIFIER(config=config_roberta, load_pretrained=True)
    # classifier.train()

    mode = 'test'
    filepath = os.path.join(cpt_path, 'epoch4_batch500_f1_score_0.8019.pth')
    time_start = time.perf_counter()
    predict_analysis(check_point=filepath,
                    mode=mode,
                    submission_file='submission/submission_0.8019.tsv')

    time_end = time.perf_counter()
    print('time: {}s'.format(time_end - time_start))