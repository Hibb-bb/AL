# -*- coding: utf-8 -*-
import json
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

CONFIG = {
    "hidden_size": (128, 128),
    "num_layers": 1,
    "bias": False,
    "batch_first": True,
    "dropout": 0.,
    "bidirectional": True,
    "vocab_size": (25000, 25000),
    "embedding_dim": (300, 128)
}


class ALComponent(nn.Module):

    x: Tensor
    y: Tensor
    loss_b: Tensor
    loss_d: Tensor
    _s: Tensor
    _t: Tensor
    _s_prime: Tensor
    _t_prime: Tensor

    def __init__(
        self,
        f: nn.Module,
        g: nn.Module,
        bx: nn.Module,
        dy: nn.Module,
        cb: nn.Module,
        ca: nn.Module,
    ) -> None:

        super(ALComponent, self).__init__()

        self.f = f
        self.g = g
        # birdge function
        self.bx = bx
        # h function
        self.dy = dy
        # loss function
        self.criterion_br = cb
        self.criterion_ae = ca

        self.dropout = nn.Dropout(CONFIG["dropout"])

    def forward(self, x, y):

        self.x = x
        self.y = y

        if self.training:

            self._s = self.f(x)
            self._t = self.g(y)
            self._t_prime = self.dy(self._t)
            return self._s.detach(), self._t.detach()

        else:

            self._s = self.f(x)
            return self._s.detach(), self._t_prime.detach()

    def loss(self):

        self.loss_b = self.criterion_br(self.bx(self._s), self._t)
        self.loss_d = self.criterion_ae(self._t_prime, self.y)

        return self.loss_b + self.loss_d


class EmbeddingAL(ALComponent):

    def __init__(
        self,
        num_embeddings: Tuple[int, int],
        embedding_dim: Tuple[int, int],
        pretrained= None,
        padding_idx: int = 0,
        lin: bool = False,
        act: nn.Module = None,
    ) -> None:

        if act == None:
            act = nn.ELU()

        if pretrained is not None:
            f = nn.Embedding.from_pretrained(
                pretrained, padding_idx=padding_idx, freeze=False
            )
        else:
            f = nn.Embedding(
                num_embeddings[0], embedding_dim[0], padding_idx=padding_idx
            )

        g = nn.Embedding(
            num_embeddings[1], embedding_dim[1], padding_idx=padding_idx
        )
        # bridge function
        bx = nn.Sequential(
            nn.Linear(embedding_dim[0], embedding_dim[1], bias=False),
            act
        )
        self.output_dim = num_embeddings[1]
        dy = nn.Sequential(
            nn.Linear(embedding_dim[1], self.output_dim, bias=False),
            act
        )
        # loss function
        cb = nn.MSELoss()
        ca = nn.CrossEntropyLoss()

        super(EmbeddingAL, self).__init__(f, g, bx, dy, cb, ca)

    def loss(self):

        p = self._s
        q = self._t

        p_nonzero = (p != 0.).sum(dim=1)
        p = p.sum(dim=1) / p_nonzero

        loss_b = self.criterion_br(self.bx(p), q)
        loss_d = self.criterion_ae(
            self._t_prime, self.y.argmax(dim=-1))

        self.loss_b = loss_b.item()
        self.loss_d = loss_d.item()

        return loss_b + loss_d

    def forward(self, x, y):

        self.x = x
        self.y = y

        if self.training:

            self._s = self.f(x)
            self._t = self.g(y.argmax(dim=-1))
            self._t_prime = self.dy(self._t)
            return self._s.detach(), self._t.detach()

        else:

            self._s = self.f(x)
            return self._s.detach(), self._t_prime.detach()


class LinearAL(ALComponent):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: Tuple[int, int],
        bias: bool = False,
    ) -> None:

        f = nn.Linear(in_features, hidden_size[0], bias=bias)
        g = nn.Linear(out_features, hidden_size[1], bias=bias)
        # bridge function
        bx = nn.Sequential(
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.Sigmoid()
        )
        # h function
        dy = nn.Sequential(
            nn.Linear(hidden_size[1], out_features),
            nn.Sigmoid()
        )
        # loss function
        cb = nn.MSELoss()
        ca = nn.MSELoss()

        super(LinearAL, self).__init__(f, g, bx, dy, cb, ca)


class LSTMAL(ALComponent):

    _h_nx: Tensor
    _h_ny: Tensor

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: Tuple[int, int],
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.,
        bidirectional: bool = False,
        act: nn.Module = None
    ) -> None:

        if act == None:
            act = nn.ELU()

        if bidirectional:
            self.d = 2
        else:
            self.d = 1

        f = nn.LSTM(
            input_size,
            hidden_size[0],
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        g = nn.Sequential(
            nn.Linear(output_size, hidden_size[1], bias=False),
            act
        )
        # bridge function
        bx = nn.Sequential(
            nn.Linear(hidden_size[0] * self.d, hidden_size[1], bias=False),
            act
        )
        # h function
        dy = nn.Sequential(
            nn.Linear(hidden_size[1], output_size, bias=False),
            act
        )
        # loss function
        cb = nn.MSELoss(reduction='mean')
        ca = nn.MSELoss(reduction='mean')

        super(LSTMAL, self).__init__(f, g, bx, dy, cb, ca)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        """
        Args:
            x: (L, N, Hin) or (N, L, Hin)
            y: (L, N, Hout) or (N, L, Hout)
            hx: ((D * num_layers, N, Hout), (D * num_layers, N, Hcell))\n
            L: sequence length
            N: batch size
            D: bidirectional\n
        Returns:
            x outputs: output x, (hx_n, cx_n)\n
            y outputs: output y, (hy_n, hy_n)\n
        """

        self.x = x
        self.y = y

        if self.training:

            self._s, (self._h_nx, c_nx) = self.f(x, hx)
            self._h_nx = self._h_nx.view(self._h_nx.size(1), -1)
            self._t = self.g(y)
            self._t_prime = self.dy(self._t)

            return self._s.detach(), (self._h_nx.detach(), c_nx.detach()), self._t.detach()

        else:

            self._s, (self._h_nx, c_nx) = self.f(x, hx)
            self._h_nx = self._h_nx.view(self._h_nx.size(1), -1)
            output = self.dy(y)

            return self._s.detach(), (self._h_nx.detach(), c_nx.detach()), output.detach()

    def loss(self):

        # last output of rnn.
        p = self._s[:, -1, :]
        q = self._t

        self.loss_b = self.criterion_br(self.bx(p), q)
        self.loss_d = self.criterion_ae(self._t_prime, self.y)

        return self.loss_b + self.loss_d


class TransformerEncoderAL(ALComponent):

    def __init__(
        self,
        d_model: Tuple[int, int],
        nhead: int,
        y_hidden: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        act: nn.Module = None,
    ) -> None:

        if act == None:
            act = nn.ELU()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model[0], nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first
        )
        # num layer = 1
        f = nn.TransformerEncoder(encoder_layer, 1)
        g = nn.Sequential(
            nn.Linear(d_model[1], y_hidden, bias=False),
            act
        )
        # bridge function
        bx = nn.Sequential(
            nn.Linear(d_model[0], y_hidden, bias=False),
            act
        )
        # h function
        dy = nn.Sequential(
            nn.Linear(y_hidden, d_model[1], bias=False),
            act
        )
        # loss function
        cb = nn.MSELoss(reduction='mean')
        ca = nn.MSELoss(reduction='mean')

        super().__init__(f, g, bx, dy, cb, ca)

    def forward(self, x, y, src_mask=None, src_key_padding_mask=None):

        self.x = x
        self.y = y

        if self.training:

            self._s = self.f(x, src_mask, src_key_padding_mask)
            self._s_prime = self.bx(self._s)
            self._t = self.g(y)
            self._t_prime = self.dy(self._t)

            return self._s.detach(), self._t.detach()

        else:

            self._s = self.f(x, src_mask, src_key_padding_mask)
            output = self.dy(y)

            return self._s.detach(), output.detach()

    def loss(self):

        p = self._s_prime
        q = self._t

        # mean
        p_nonzero = (p != 0.).sum(dim=1)
        p = p.sum(dim=1) / p_nonzero

        self.loss_b = self.criterion_br(p, q)
        self.loss_d = self.criterion_ae(self._t_prime, self.y)

        return self.loss_b + self.loss_d

    def _generate_square_subsequent_mask(self, sz: int):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).

        Shape: (sz, sz).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def load_parameters():

    global CONFIG
    with open("configs/hyperparameters.json", "r", encoding="utf8") as f:
        CONFIG = json.load(f)


def save_parameters():

    with open("configs/hyperparameters.json", "w", encoding="utf8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, sort_keys=True, indent=3)


if __name__ == "__main__":
    save_parameters()
