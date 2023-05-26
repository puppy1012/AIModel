import torch
from torch import nn
import mxnet
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import os

# kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

import pandas as pd
# transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


# GPU 사용

device = torch.device("cuda:0")
print("cuda: ", torch.cuda.is_available())

# BERT 모델, Vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=5,  ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 100
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

PATH = 'C:\\Users\\15\\Desktop\\kobert\\KoBERT\\pt'
model = torch.load(PATH + '\\KoBERT_담화.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load(PATH + '\\model_state_dict.pt'))

# 토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]
    output_data = 0
    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                output_data = 3
                test_eval.append("행복")
                #output_data = [0, 0, 0, 1, 0]

            elif np.argmax(logits) == 1:
                output_data = 4
                test_eval.append("슬픔")
                #output_data = [0, 0, 0, 0, 1]
            elif np.argmax(logits) == 2:
                output_data = 0
                test_eval.append("분노")
                #output_data = [1, 0, 0, 0, 0]
            elif np.argmax(logits) == 3:
                output_data = 1
                test_eval.append("놀람")
                #output_data = [0, 1, 0, 0, 0]
            elif np.argmax(logits) == 4:
                output_data = 3
                test_eval.append("중립")
                #output_data = [0, 0, 1, 0, 0]
        print(output_data, ", ", test_eval[0])

# while(True):
#     sentence = input("입력: ")
#     predict(sentence)

with open('./동화-1.txt' , "r", encoding="utf-8") as f:
    example = f.read()
    example = example.replace('?', '.')
    example = example.replace('!', '.')
    example = example.replace('\'', '')
    strings = example.split('.')
    strings.pop()
for string in strings:
    print(string)
    emotion = predict(string)
