import torch
import torch.nn as nn
from transformer.encoder import TransformerEncoder

class AE(nn.Module):
    def __init__(self, inp_dim, out_dim, cri='ce'):
        super().__init__()

        self.g = nn.Sequential(
            nn.Linear(inp_dim, out_dim),
            nn.Tanh()            
        )

        if cri == 'ce':
            self.h = nn.Sequential(
                nn.Linear(out_dim, inp_dim),
                nn.Tanh()            
            )
        else:
            self.h = nn.Sequential(
                nn.Linear(out_dim, inp_dim),
            )

        if cri == 'ce':
            self.cri = nn.CrossEntropyLoss()
        else:
            self.cri = nn.MSELoss()
        self.mode = cri
    
    def forward(self, x):
        enc_x = self.g(x)
        rec_x = self.h(enc_x)
        if self.mode == 'ce':
            return enc_x, self.cri(rec_x, x.argmax(1))
        else:
            return enc_x, self.cri(rec_x, x)

class ENC(nn.Module):
    def __init__(self, inp_dim, out_dim, lab_dim=128, f='emb', n_heads=4, word_vec=None):
        super().__init__()

        self.b = nn.Sequential(
            nn.Linear(out_dim, lab_dim),
            nn.Tanh()
        )
        self.mode = f
        if f == 'emb':
            self.f = nn.Embedding(inp_dim, out_dim)
            if word_vec is not None:
                self.f = nn.Embedding.from_pretrained(word_vec, freeze=False)
        elif f == 'lstm':
            self.f = nn.LSTM(inp_dim, out_dim, bidirectional=True, batch_first=True)
        elif f == 'trans':
            self.f = TransformerEncoder(d_model=inp_dim, d_ff=out_dim, n_heads=n_heads)
            self.b = nn.Sequential(
                nn.Linear(inp_dim, lab_dim),
                nn.Tanh()
            )

        self.cri = nn.MSELoss()
    
    def forward(self, x, tgt, mask=None, h=None):

        if self.mode == 'emb':
            enc_x = self.f(x)
        elif self.mode == 'lstm':
            enc_x, (h, c) = self.f(x, h)
        elif self.mode == 'trans':
            enc_x = self.f(x, mask=mask)
        
        red_x = self.reduction(enc_x, mask, h)
        red_x = self.b(red_x)
        loss = self.cri(red_x, tgt)
        return enc_x, loss, h, mask

    def reduction(self, x, mask=None, h=None):

        # to match bridge function
        if self.mode == 'emb':
            return x.mean(1)

        elif self.mode == 'lstm':
            _h = h[0] + h[1]
            return _h

        elif self.mode == 'trans':

            denom = torch.sum(mask, -1, keepdim=True)
            feat = torch.sum(x * mask.unsqueeze(-1), dim=1) / denom
            return feat

class TransLayer(nn.Module):
    def __init__(self, inp_dim, lab_dim, hid_dim, lr):
        super().__init__()

        self.enc = ENC(inp_dim, hid_dim, lab_dim=lab_dim, f='trans')
        self.ae = AE(lab_dim, lab_dim, cri='mse')

        self.ae_opt = torch.optim.Adam(self.ae.parameters(), lr=0.0005)
        self.enc_opt = torch.optim.Adam(self.enc.parameters(), lr=lr)
    
    def forward(self, x, y, mask):

        self.ae_opt.zero_grad()
        enc_y , ae_loss = self.ae(y)
        ae_loss.backward()
        self.ae_opt.step()
    
        self.enc_opt.zero_grad()
        tgt = enc_y.clone().detach()
        enc_x, enc_loss, h, mask = self.enc(x, tgt, mask=mask)
        enc_loss.backward()
        self.enc_opt.step()

        return enc_x.detach(), enc_y.detach(), ae_loss, enc_loss, mask        


class TransModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, l1_dim, lr, class_num, lab_dim=128, word_vec=None):
        super().__init__()

        self.emb = EMBLayer(vocab_size, lab_dim, emb_dim, lr = 0.001, class_num=class_num, word_vec=word_vec)
        self.l1 = TransLayer(emb_dim, lab_dim, l1_dim, lr=lr)
        self.l1_dim = l1_dim
        self.l2 = TransLayer(emb_dim, lab_dim, l1_dim, lr=lr)
        self.losses = [0.0] * 6
        self.class_num = class_num
    def forward(self, x, y):

        mask = self.get_mask(x)
        y = torch.nn.functional.one_hot(y, self.class_num).float().to(y.device)
        emb_x, emb_y, emb_ae, emb_as, _ = self.emb(x, y) # also updated

        l1_x, l1_y, l1_ae, l1_as, mask = self.l1(emb_x, emb_y, mask)
        l2_x, l2_y, l2_ae, l2_as, mask = self.l2(l1_x, l1_y, mask)

        return [emb_ae, emb_as, l1_ae, l1_as, l2_ae, l2_as]
    
    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask.cuda()

    def inference(self, x):

        mask = self.get_mask(x)
        emb_x = self.emb.enc.f(x)
        l1_x = self.l1.enc.f(emb_x, mask)
        l2_x = self.l2.enc.f(l1_x, mask)

        denom = torch.sum(mask, -1, keepdim=True)
        feat = torch.sum(l2_x * mask.unsqueeze(-1), dim=1) / denom
        bridge = self.l2.enc.b(feat)

        _out = self.l2.ae.h(bridge)
        _out = self.l1.ae.h(_out)
        pred = self.emb.ae.h(_out)

        return pred

class LSTMLayer(nn.Module):
    def __init__(self, inp_dim, lab_dim, hid_dim, lr):
        super().__init__()

        self.enc = ENC(inp_dim, hid_dim, lab_dim=lab_dim, f='lstm')
        self.ae = AE(lab_dim, lab_dim, cri='mse')
    
        self.ae_opt = torch.optim.Adam(self.ae.parameters(), lr=lr)
        self.enc_opt = torch.optim.Adam(self.enc.parameters(), lr=lr)

    def forward(self, x, y, mask=None, h=None):

        self.ae_opt.zero_grad()
        enc_y , ae_loss = self.ae(y)
        ae_loss.backward()
        self.ae_opt.step()
    
        self.enc_opt.zero_grad()
        tgt = enc_y.clone().detach()
        enc_x, enc_loss, hidden, _ = self.enc(x, tgt, mask, h)
        enc_loss.backward()
        self.enc_opt.step()
        (h, c) = hidden
        h = h.reshape(2, x.size(0), -1)
        hidden = (h.detach(), c.detach())

        return enc_x.detach(), enc_y.detach(), ae_loss, enc_loss, [hidden, mask]

class EMBLayer(nn.Module):
    def __init__(self, inp_dim, lab_dim, hid_dim, lr, class_num=None, word_vec=None):
        super().__init__()

        self.enc = ENC(inp_dim, hid_dim, lab_dim=lab_dim, f='emb', word_vec=word_vec)
        assert class_num is not None
        self.ae = AE(class_num, lab_dim, cri='ce')
    
        self.ae_opt = torch.optim.Adam(self.ae.parameters(), lr=lr)
        self.enc_opt = torch.optim.Adam(self.enc.parameters(), lr=lr)

    def forward(self, x, y, mask=None, h=None):

        self.ae_opt.zero_grad()
        enc_y , ae_loss = self.ae(y)
        ae_loss.backward()
        self.ae_opt.step()
    
        self.enc_opt.zero_grad()
        tgt = enc_y.clone().detach()
        enc_x, enc_loss, hidden, mask = self.enc(x, tgt, mask, h)
        enc_loss.backward()
        self.enc_opt.step()

        return enc_x.detach(), enc_y.detach(), ae_loss, enc_loss, [hidden, mask]

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, l1_dim, lr, class_num, lab_dim=128, word_vec=None):
        super().__init__()

        self.emb = EMBLayer(vocab_size, lab_dim, emb_dim, lr = 0.001, class_num=class_num, word_vec=word_vec)
        self.l1 = LSTMLayer(emb_dim, lab_dim, l1_dim, lr=lr)
        self.l1_dim = l1_dim
        self.l2 = LSTMLayer(l1_dim*2, lab_dim, l1_dim, lr=lr)
        self.losses = [0.0] * 6
        self.class_num = class_num

    def forward(self, x, y):
        
        y = torch.nn.functional.one_hot(y, self.class_num).float().to(y.device)
        emb_x, emb_y, emb_ae, emb_as, _ = self.emb(x, y) # also updated

        l1_x, l1_y, l1_ae, l1_as, [h, _] = self.l1(emb_x, emb_y)
        l1_x = torch.cat((l1_x[:, :, :self.l1_dim], l1_x[:, :, self.l1_dim:]), dim=-1)

        l2_x, l2_y, l2_ae, l2_as, [h, _] = self.l2(l1_x, l1_y, h)

        return [emb_ae, emb_as, l1_ae, l1_as, l2_ae, l2_as]
    
    def inference(self, x):
        emb_x = self.emb.enc.f(x)
        l1_x, (h, c) = self.l1.enc.f(emb_x)
        l1_x = torch.cat((l1_x[:, :, :self.l1_dim], l1_x[:, :, self.l1_dim:]), dim=-1)
        # print(l1_x.shape, )
        l2_x, (h, c) = self.l2.enc.f(l1_x, (h, c))
        h = h[0] + h[1]

        bridge = self.l2.enc.b(h)

        _out = self.l2.ae.h(bridge)
        _out = self.l1.ae.h(_out)
        pred = self.emb.ae.h(_out)

        return pred


# model = LSTMModel(vocab_size=1000, emb_dim=40, l1_dim=40, lr=0.001, class_num=4)

# x = torch.randint(0, 900, size=(64, 15))
# y = torch.randint(0, 3, size=(64, ))

# losses = model(x, y)

# pred = model.inference(x)

# model = TransModel(vocab_size=1000, emb_dim=40, l1_dim=40, lr=0.001, class_num=4)

# x = torch.randint(0, 900, size=(64, 15))
# mask = torch.ones_like(x)
# y = torch.randint(0, 3, size=(64, ))

# losses = model(x, y, mask)

# pred = model.inference(x, mask)
