# -*- coding: utf-8 -*-
import itertools
import math
import re
import string
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from prettytable import PrettyTable
import io
from tqdm import tqdm

import nltk
nltk.download('stopwords')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, src: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        pe = self.pe.detach().to(src.device)
        output = src + pe[:, :src.size(1)]
        return self.dropout(output)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def get_act(args):
    if args.act == 'tanh':
        act = nn.Tanh()
    elif args.act == 'sigmoid':
        act = nn.Sigmoid()
    elif args.act == 'relu':
        act = nn.ReLU()
    elif args.act == 'relu6':
        act = nn.ReLU6()
    elif args.act == 'leakyrelu':
        act = nn.LeakyReLU()
    elif args.act == 'elu':
        act = nn.ELU()
    elif args.act == 'gelu':
        act = nn.GELU()
    elif args.act == 'prelu':
        act = nn.PReLU()
    elif args.act == 'selu':
        act = nn.SELU()
    elif args.act == 'tanhsh':
        act = nn.Tanhshrink()
    else:
        act = None
    return act

def get_word_vector(vocab, emb='glove'):

    if emb == 'glove':
        fname = 'associated_learning/glove.6B.300d.txt'
        
        with open(fname,'rt') as fi:
            full_content = fi.read().strip().split('\n')

        data = {}
        for i in tqdm(range(len(full_content)), total=len(full_content), desc = 'loading glove vocabs...'):
            i_word = full_content[i].split(' ')[0]
            if i_word not in vocab.keys():
                continue
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            data[i_word] = i_embeddings

    elif emb == 'fasttext':
        fname = 'associated_learning/wiki-news-300d-1M.vec'

        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}

        for line in tqdm(fin, total=1000000, desc='loading fasttext vocabs...'):
            tokens = line.rstrip().split(' ')
            if tokens[0] not in vocab.keys():
                continue
            data[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
    
    else:
        raise Exception('emb not implemented')

    w = []
    find = 0
    for word in vocab.keys():
        try:
            w.append(torch.tensor(data[word]))
            find += 1
        except:
            w.append(torch.rand(300))

    print('found', find, 'words in', emb)
    return torch.stack(w, dim=0)

def data_preprocessing(text, remove_stopword=False):

    stop_words = set(stopwords.words('english'))

    text = text.lower()
    # text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub('<.*?>', '', text)
    text = ''.join([c for c in text if c not in string.punctuation])

    if remove_stopword:
        text = [word for word in text.split() if word not in stop_words]
    else:
        text = [word for word in text.split()]

    text = ' '.join(text)

    return text


def create_vocab(corpus, vocab_size=30000):

    corpus = [t.split() for t in corpus]
    corpus = list(itertools.chain.from_iterable(corpus))
    count_words = Counter(corpus)
    print('total count words', len(count_words))
    sorted_words = count_words.most_common()

    if vocab_size > len(sorted_words):
        v = len(sorted_words)
    else:
        v = vocab_size - 2

    vocab_to_int = {w: i + 2 for i, (w, c) in enumerate(sorted_words[:v])}

    vocab_to_int['<pad>'] = 0
    vocab_to_int['<unk>'] = 1
    print('vocab size', len(vocab_to_int))

    return vocab_to_int


def multi_class_process(labels, label_num):
    '''
    this function will convert multi-label lists into one-hot vector or n-hot vector
    '''
    hot_vecs = []
    for l in labels:
        b = torch.zeros(label_num)
        b[l] = 1
        hot_vecs.append(b)
    return hot_vecs


def multi_doubleclass_process(labels, label_num):
    '''
    this function will convert multi-label lists into one-hot vector or n-hot vector
    '''
    hot_vecs = []
    for l in labels:
        b = torch.zeros(label_num*2)
        b[l] = 1
        b[l+label_num] = 1
        hot_vecs.append(b)
    return hot_vecs


def multi_label_process(labels, label_num):
    '''
    this function will convert multi-label lists into one-hot vector or n-hot vector
    '''
    hot_vecs = []
    for l in labels:
        b = torch.zeros(label_num)
        for i in l:
            b[i] = 1
        hot_vecs.append(b)
    return hot_vecs


def convert2id(corpus, vocab_to_int):
    '''
    a list of string
    '''
    reviews_int = []
    for text in corpus:
        r = []
        for word in text.split():
            if word in vocab_to_int.keys():
                r.append(vocab_to_int[word])
            else:
                r.append(vocab_to_int['<unk>'])
        reviews_int.append(r)
    return reviews_int


def Padding(review_int, seq_len):
    '''
    Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(review_int), seq_len), dtype=int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)

    return features


def PadTransformer(review_int, seq_len):
    '''
    Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    masks = np.zeros((len(review_int), seq_len), dtype=bool)
    features = np.zeros((len(review_int), seq_len), dtype=int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
            p1 = np.ones(seq_len-len(review), dtype=bool)
            p2 = np.zeros(len(review), dtype=bool)
            new_mask = np.concatenate((p1, p2), axis=0)
        else:
            new = review[: seq_len]
            new_mask = np.zeros(seq_len, dtype=bool)
        features[i, :] = np.array(new)
        masks[i, :] = new_mask
    return features, masks
