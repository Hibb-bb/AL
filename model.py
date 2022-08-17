import torch
import torch.nn as nn



class TranAL(nn.Module):
    def __init__(self, inp_dim, hid_dim, lab_dim):
        super().__init__()

        self.f = nn.TransformerEncoderLayer(d_model=inp_dim, dim_feedforward=2048, nhead=6, batch_first=True)
        self.g = nn.Sequential(
                nn.Linear(lab_dim, lab_dim),
                nn.Tanh()
                )
        self.h = nn.Sequential(
                nn.Linear(lab_dim, lab_dim),
                nn.Tanh()
                )
        self.b = nn.Sequential(
                nn.Linear(inp_dim, lab_dim),
                nn.Tanh()
                )
        self.cri = nn.MSELoss()

    def forward(self, x, y, mask=None):
        if mask is None:
            mask = self.get_mask(x)
        enc_x = self.f(x, src_key_padding_mask=mask)
        enc_y = self.g(y)
        
        p_nonzero = (enc_x != 0.).sum(dim=1)
        _x = enc_x.sum(dim=1) / p_nonzero

        b_x = self.b(_x)
        ae_y = self.h(enc_y)
        
        return self.cri(ae_y, y) + self.cri(b_x, enc_y.detach()), enc_x.detach(), enc_y.detach()

    def inference(self, x, mask=None, bridge=False):
        if mask is None:
            mask = self.get_mask(x)
        if not bridge:
            return self.f(x, src_key_padding_mask=mask)
        else:
            _x = self.f(x, src_key_padding_mask=mask)
            p_nonzero = (_x != 0.).sum(dim=1)
            _x = _x.sum(dim=1) / p_nonzero
            return self.b(_x)

    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask.cuda()

class EMBAL(nn.Module):
    def __init__(self, inp_dim, hid_dim, class_num, lab_dim, pre_emb):
        super().__init__()

        self.f = nn.Embedding(inp_dim, hid_dim)
        if pre_emb is not None:
            self.f = nn.Embedding.from_pretrained(pre_emb, freeze=False)
        self.g = nn.Embedding(class_num, lab_dim)
        self.h = nn.Sequential(
            nn.Linear(lab_dim, class_num)
            )
        self.b = nn.Sequential(
            nn.Linear(hid_dim, lab_dim)
            )
        self.act = nn.Tanh()
        self.dp = nn.Dropout(0.3)
        self.ae_cri = nn.CrossEntropyLoss()
        self.as_cri = nn.MSELoss()
        
    def forward(self, x, y):

        enc_x = self.dp(self.f(x))
        _x = enc_x.mean(1)
        enc_y = self.dp(self.act(self.g(y)))
        b_x = self.act(self.b(_x))
        ae_y = self.act(self.h(enc_y))
        return self.ae_cri(ae_y, y) + self.as_cri(b_x, enc_y.detach()), enc_x.detach(), enc_y.detach()
        
    def inference(self, x):
        return self.f(x)

class LSTMAL(nn.Module):
    def __init__(self, inp_dim, hid_dim, lab_dim):
        super().__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.f = nn.LSTM(inp_dim, hid_dim, bidirectional=True, batch_first=True)
        self.g = nn.Sequential(
            nn.Linear(lab_dim, lab_dim),
            nn.Tanh()
            )
        self.h = nn.Sequential(
            nn.Linear(lab_dim, lab_dim),
            nn.Tanh()
            )
        self.b = nn.Sequential(
            nn.Linear(hid_dim, lab_dim),
            nn.Tanh()
            )
        self.dp = nn.Dropout(0.3)
        self.cri = nn.MSELoss()

    def forward(self, x, y, hidden=None):

        if hidden is None:
            enc_x, (h, c) = self.f(x)
        else:
            enc_x, (h, c) = self.f(x, hidden)
        
        _h =self.dp(h[0] + h[1])
        enc_x = self.dp(enc_x)
        enc_y = self.dp(self.g(y))

        b_x = self.b(_h)
        ae_y = self.h(enc_y)
        h = h.reshape(2, x.size(0), -1)

        return self.cri(ae_y, y) + self.cri(b_x, enc_y.detach()), enc_x.detach(),  (h.detach(), c.detach()), enc_y.detach()
    
    def inference(self, x, hidden=None, to_b=False):

        if hidden is None:
            enc_x, (h, c) = self.f(x)
        else:
            enc_x, (h, c) = self.f(x, hidden)
            _h = h[0] + h[1]

        if to_b:
            return self.b(_h)
        else:
            return enc_x, (h, c)

class Model(nn.Module):
    def __init__(self, emb_dim, l1_dim, l2_dim, class_num, vocab_size, word_vec=None):
        super().__init__()

        self.l1_dim = l1_dim
        self.emb = EMBAL(vocab_size, emb_dim, class_num, 128, word_vec)
        self.lstm1 = LSTMAL(emb_dim, l1_dim, 128)
        self.lstm2 = LSTMAL(l1_dim*2, l2_dim, 128)
        self.opts = [torch.optim.Adam(self.emb.parameters()), torch.optim.Adam(self.lstm1.parameters()), torch.optim.Adam(self.lstm2.parameters())]

    def forward(self, x, y):

        self.opts[0].zero_grad()
        emb_loss, emb_x, emb_y = self.emb(x, y)
        emb_loss.backward()
        self.opts[0].step()

        self.opts[1].zero_grad()
        l1_loss, emb_x, h, emb_y = self.lstm1(emb_x, emb_y)
        l1_loss.backward()
        self.opts[1].step()
        emb_x = torch.cat((emb_x[:, :, :self.l1_dim], emb_x[:, :, self.l1_dim:]), dim=-1)

        self.opts[2].zero_grad()
        l2_loss, emb_x, h, emb_y = self.lstm2(emb_x, emb_y, h)
        l2_loss.backward()
        self.opts[2].step()

        return emb_loss + l1_loss + l2_loss

    def inference(self, x):

        emb_x = self.emb.inference(x)
        l1_x, hiddden = self.lstm1.inference(emb_x)
        l1_x = torch.cat((l1_x[:, :, :self.l1_dim], l1_x[:, :, self.l1_dim:]), dim=-1)
        bridge = self.lstm2.inference(l1_x, hiddden, True)

        _out = self.lstm2.h(bridge)
        _out = self.lstm1.h(_out)
        pred = self.emb.h(_out)

        return pred

class TranModel(nn.Module):
    def __init__(self, class_num, vocab_size, lr, pre_emb=None):
        super().__init__()

        self.emb = EMBAL(vocab_size, 300, class_num, 128, pre_emb)
        self.l1 = TranAL(300, 300, 128)
        self.l2 = TranAL(300, 300, 128)
        self.opts = [torch.optim.Adam(self.emb.parameters(), lr=lr), torch.optim.Adam(self.l1.parameters(), lr=lr), torch.optim.Adam(self.l2.parameters(), lr=lr)]

    def forward(self, x, y):

        self.opts[0].zero_grad()
        emb_loss, emb_x, emb_y = self.emb(x, y)
        emb_loss.backward()
        self.opts[0].step()

        self.opts[1].zero_grad()
        mask = self.get_mask(x)
        l1_loss, emb_x, emb_y = self.l1(emb_x, emb_y, mask)
        l1_loss.backward()
        self.opts[1].step()

        self.opts[2].zero_grad()
        l2_loss, emb_x, emb_y = self.l2(emb_x, emb_y, mask)
        l2_loss.backward()
        self.opts[2].step()

        return emb_loss + l1_loss + l2_loss

    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask.cuda()

    def inference(self, x):

        emb_x = self.emb.inference(x)
        mask = self.get_mask(x)
        l1_x = self.l2.inference(emb_x, mask)
        l2_x = self.l2.inference(l1_x, mask, bridge=True)
        _out = self.l2.h(l2_x)
        _out = self.l1.h(_out)
        pred = self.emb.h(_out)

        return pred