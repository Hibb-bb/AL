#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# import seaborn as sns

import re
import string
from collections import Counter
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from classification.model import EmbeddingAL, LSTMAL

import sys

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

df = pd.read_csv('IMDB_Dataset.csv')


def data_preprocessing(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text) # Remove HTML from text
    text = ''.join([c for c in text if c not in string.punctuation])# Remove punctuation
    text = [word for word in text.split() if word not in stop_words]
    text = ' '.join(text)
    return text


df['cleaned_reviews'] = df['review'].apply(data_preprocessing)
corpus = [word for text in df['cleaned_reviews'] for word in text.split()]
count_words = Counter(corpus)
sorted_words = count_words.most_common()


vocab_to_int = {w:i+2 for i, (w,c) in enumerate(sorted_words[:29999])}

vocab_to_int['pad'] = 0
vocab_to_int['unk'] = 1


print('vocab size',len(vocab_to_int))

reviews_int = []
for text in df['cleaned_reviews']:
    r=[]
    for word in text.split():
        try:
            r.append(vocab_to_int[word])
        except:
            r.append(vocab_to_int['unk'])
    # r = [vocab_to_int[word] for word in text.split() if ]
    reviews_int.append(r)

# print(reviews_int[:1])
df['Review int'] = reviews_int


df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)


review_len = [len(x) for x in reviews_int]
df['Review len'] = review_len

# print(df['Review len'].describe())

# df['Review len'].hist()


def Padding(review_int, seq_len):
    '''
    Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_len), dtype = int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)
            
    return features


features = Padding(reviews_int, 200)

X_train, X_remain, y_train, y_remain = train_test_split(features, df['sentiment'].to_numpy(), test_size=0.2, random_state=1)
X_valid, X_test, y_valid, y_test = train_test_split(X_remain, y_remain, test_size=0.5, random_state=1)

# create tensor dataset
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
valid_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))

# dataloaders
batch_size = 128

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)


# In[15]:


# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

save_dir = sys.argv[1] + '.pt'

class SentAL(nn.Module):
    def __init__(self, emb, l1, l2):
        super(SentAL, self).__init__()
        self.embedding = emb
        self.layer_1 = l1
        self.layer_2 = l2
        self.dropout = nn.Dropout(0.1)
    def forward(self, x, y):
        emb_x, emb_y, = self.embedding(x, y)
        emb_x, emb_y = self.dropout(emb_x), self.dropout(emb_y)
        # print(self.embedding._t_prime.shape, self.embedding.y.shape)
        emb_loss = self.embedding.loss() 

        layer_1_x, h1 , layer_1_y = self.layer_1(emb_x, emb_y)
        layer_1_x, layer_1_y = self.dropout(layer_1_x), self.dropout(layer_1_y)
        layer_1_loss = self.layer_1.loss()

        h,c = h1
        h = h
        c = c
        # print(h.shape, c.shape)
        h = h.reshape(2, h.size(0), -1)
        h1 = (h,c)
        
        layer_2_x, h2, layer_2_y = self.layer_2(layer_1_x, layer_1_y, h1)
        layer_2_loss = self.layer_2.loss()

        return emb_loss, layer_1_loss, layer_2_loss


torch.cuda.empty_cache()

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# device = 'cpu'

# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int)+1
output_size = 1
embedding_dim = 300
hidden_dim = 400
n_layers = 2

emb = EmbeddingAL((vocab_size, 2), (300, 128))
l1 = LSTMAL(300, 128, (128,128), dropout=0.1, bidirectional=True)
l2 = LSTMAL(256, 128, (128,128), dropout=0.1, bidirectional=True)

model = SentAL(emb, l1, l2)
model = model.to(device)
print('model param', get_n_params(model))
# raise Exception('ok')

lr=0.001

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer_1 = torch.optim.Adam(model.embedding.parameters(), lr=1e-4)
# optimizer_2 = torch.optim.Adam(model.layer_1.parameters(), lr=1e-4)
# optimizer_3 = torch.optim.Adam(model.layer_2.parameters(), lr=1e-4)
# function to predict accuracy
def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

clip = 5
epochs = 20
valid_acc_min = 0
# train for some number of epochs
epoch_tr_loss,epoch_vl_loss = [],[]
epoch_tr_acc,epoch_vl_acc = [],[]

for epoch in range(epochs):
    train_losses = []

    total_emb_loss = []
    total_l1_loss = []
    total_l2_loss = []

    total_acc = 0.0
    total_count = 0
    model.embedding.train()
    model.layer_1.train()
    model.layer_2.train()
    model.embedding.training=True
    model.layer_1.training=True
    model.layer_2.training=True

    # initialize hidden state 
    for inputs, labels in train_loader:
        model.train()

        inputs, labels = inputs.to(device), labels.to(device)   
        
        optimizer.zero_grad()

        emb_loss, l1_loss, l2_loss = model(inputs,labels)
        
        nn.utils.clip_grad_norm_(model.layer_1.parameters(), clip)
        nn.utils.clip_grad_norm_(model.layer_2.parameters(), clip)
        
        loss = emb_loss + l1_loss + l2_loss
        loss.backward()

        total_emb_loss.append(emb_loss.item())
        total_l1_loss.append(l1_loss.item())
        total_l2_loss.append(l2_loss.item())

        optimizer.step()

        torch.cuda.empty_cache()

        # calculate the loss and perform backprop
        with torch.no_grad():
            
            model.eval()

            left  =  model.embedding.f(inputs)
            output, hidden = model.layer_1.f(left)
            # output, (left, c) = model.layer_2.f(output, hidden)
            left, (output, c) = model.layer_2.f(output, hidden)
            left = left[:,-1,:]
            # left = left.reshape(left.size(1), -1)

            right = model.layer_2.bx(left)
            right = model.layer_2.dy(right)
            right = model.layer_1.dy(right)
            predicted_label = torch.round(model.embedding.dy(right).squeeze())
            total_acc += (predicted_label == labels.to(torch.float)).sum().item()
            total_count += labels.size(0)

        # clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.layer_1.parameters(), clip)
        nn.utils.clip_grad_norm_(model.layer_2.parameters(), clip)
 
    
    val_losses = []
    val_acc = 0.0
    val_count = 0

    model.embedding.eval()
    model.layer_1.eval()
    model.layer_2.eval()

    for inputs, labels in valid_loader:
        with torch.no_grad():
            model.embedding.eval()
            model.layer_1.eval()
            model.layer_2.eval()

            inputs, labels = inputs.to(device), labels.to(device)

            left = model.embedding.f(inputs)
            output, hidden = model.layer_1.f(left)

            left, (output, c) = model.layer_2.f(output, hidden)
            left = left[:,-1,:]
            
            right = model.layer_2.bx(left)
            right = model.layer_2.dy(right)
            right = model.layer_1.dy(right)
            predicted_label = torch.round(model.embedding.dy(right).squeeze())
            val_acc += (predicted_label == labels.to(torch.float)).sum().item()
            val_count += labels.size(0)
            
    epoch_train_loss = [np.mean(total_emb_loss), np.mean(total_l1_loss), np.mean(total_l2_loss)]
    epoch_train_acc = total_acc/total_count
    epoch_val_acc = val_acc/val_count

    print(f'Epoch {epoch+1}') 
    print(f'train_loss : {epoch_train_loss}')
    print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
    if epoch_val_acc >= valid_acc_min:
        torch.save(model.state_dict(), f'{save_dir}')
        print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_acc_min,epoch_val_acc))
        valid_acc_min = epoch_val_acc
    print(25*'==')

test_losses = [] # track loss
num_correct = 0

model.eval()
model.load_state_dict(torch.load(f'{save_dir}'))
model = model.to(device)

test_acc = 0
test_count = 0

for inputs, labels in test_loader:
    with torch.no_grad():
        model.embedding.eval()
        model.layer_1.eval()
        model.layer_2.eval()

        inputs, labels = inputs.to(device), labels.to(device)

        left = model.embedding.f(inputs)
        output, hidden = model.layer_1.f(left)
        left, (output, c) = model.layer_2.f(output, hidden)
        left = left[:,-1,:]
        # left = left.reshape(left.size(1), -1)
        right = model.layer_2.bx(left)
        right = model.layer_2.dy(right)
        right = model.layer_1.dy(right)
        predicted_label = torch.round(model.embedding.dy(right).squeeze())
        test_acc += (predicted_label == labels.to(torch.float)).sum().item()
        test_count += labels.size(0)

print('valid acc', valid_acc_min)
print('Test acc', test_acc/test_count)

# -- stats! -- ##
# avg test loss
# print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = test_acc/test_count
print("Test accuracy: {:.3f}".format(test_acc))

