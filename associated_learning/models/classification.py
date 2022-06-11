# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import Tensor

from ..utils import PositionalEncoding


class LSTMForCLS(nn.Module):

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, n_layers, class_num, drop_prob=0.1, pretrain=None
    ):

        super(LSTMForCLS, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        if pretrain == None:
            self.embedding = nn.Embedding.from_pretrain(pretrain, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        n_layers=1
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 400),
            nn.ReLU(),
            nn.Linear(400, class_num)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds)
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.softmax(out)
        return out, hidden


class LSTMALForCLS(nn.Module):

    def __init__(self, emb, l1, l2):

        super(LSTMALForCLS, self).__init__()

        self.embedding = emb
        self.layer_1 = l1
        self.layer_2 = l2
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y):
        
        batch_size = x.size(0)
        direction = 2

        emb_x, emb_y, = self.embedding(x, y)

        emb_x, emb_y = self.dropout(emb_x), self.dropout(emb_y)
        emb_loss = self.embedding.loss()

        layer_1_x, h1, layer_1_y = self.layer_1(emb_x.detach(), emb_y.detach())
        layer_1_x, layer_1_y = self.dropout(layer_1_x), self.dropout(layer_1_y)

        layer_1_loss = self.layer_1.loss()

        h, c = h1
        h = h.reshape(direction, batch_size, -1)
        h1 = (h.detach(), c.detach())

        layer_2_x, h2, layer_2_y = self.layer_2(
            layer_1_x.detach(), layer_1_y.detach(), h1)

        layer_2_loss = self.layer_2.loss()

        return emb_loss, layer_1_loss, 2*layer_2_loss

    def short_cut_emb(self, x):

        left = self.embedding.f(x)
        left = left.mean(-1)
        right = self.embedding.bx(left)
        right = self.embedding.dy(right)
        return right

    def short_cut_lstm(self, x):

        left = self.embedding.f(x)
        left, hidden = self.layer_1.f(left)
        left = left[:,-1,:]
        right = self.layer_1.bx(left)
        right = self.layer_1.dy(right)
        right = self.embedding.dy(right)
        return right


class TransformerForCLS(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        nhead,
        nlayers,
        class_num,
        dropout: float = 0.5,
        pretrained: Tensor = None
    ):

        super(TransformerForCLS, self).__init__()

        self.embedding_dim = embedding_dim

        if pretrained == None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained, freeze=False, padding_idx=0
            )

        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim, nhead, hidden_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)

        decoder_layer = nn.TransformerDecoderLayer(
            embedding_dim, nhead, hidden_dim, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, nlayers)

        self.fc = nn.Linear(embedding_dim, class_num)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):

        x = self.embedding(x)
        # positional encoding.
        # x = x * math.sqrt(self.embedding_dim)
        # x = self.pos_encoder(x)

        output = self.encoder(x, src_mask, src_key_padding_mask)
        output = output.sum(dim=1)

        src_len = (src_key_padding_mask == 0).sum(dim=1)
        src_len = torch.stack((src_len,) * output.size(1), dim=1)
        output = output / src_len
        output = self.fc(output)

        return self.softmax(output)


class TransformerALForCLS(nn.Module):

    def __init__(self, embedding, l1, l2):

        super(TransformerALForCLS, self).__init__()

        self.embedding = embedding
        self.layer_1 = l1
        self.layer_2 = l2

        # NOTE: still has some bugs.
        # self.layers = nn.ModuleList([copy.deepcopy(module) for _ in range(nlayers - 1)])

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y, src_mask=None, src_key_padding_mask=None):

        layer_loss = []

        emb_x, emb_y = self.embedding(x, y)
        emb_x, emb_y = self.dropout(emb_x), self.dropout(emb_y)
        emb_loss = self.embedding.loss()

        out_x, out_y = self.layer_1(
            emb_x, emb_y, src_mask, src_key_padding_mask
        )
        layer_loss.append(self.layer_1.loss())

        out_x, out_y = self.layer_2(
            out_x, out_y, src_mask, src_key_padding_mask
        )
        layer_loss.append(self.layer_2.loss())

        return emb_loss, layer_loss

    def short_cut_emb(self, x):

        x = self.embedding.f(x)
        p_nonzero = (x != 0.).sum(dim=1)
        left = x.sum(dim=1) / p_nonzero
        right = self.embedding.bx(left)
        right = self.embedding.dy(right)

        return right

    def short_cut_l1(self, x, masks):

        left = self.embedding.f(x)
        left = self.layer_1.f(
            left, None, masks)
        left = left.sum(dim=1)
        src_len = (masks == 0).sum(dim=1)
        src_len = torch.stack((src_len,) * left.size(1), dim=1)
        left = left / src_len
        right = self.layer_1.bx(left)
        right = self.layer_1.dy(right)
        right = self.embedding.dy(right)

        return right
