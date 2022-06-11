# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from .mi_tool import MI_Vis
from .vis import tsne


class TransfomerTrainer:

    def __init__(
        self, model, lr, train_loader, valid_loader, test_loader, save_dir, label_num=None, double=False, class_num=None, is_al=False
    ):

        self.model = model
        project_name = save_dir.replace('/', '-')
        wandb.login(key='token id')
        wandb.init(project=project_name, entity='team')
        wandb.watch(self.model)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.opt = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.cri = nn.NLLLoss()
        self.clip = 5

        self.label_num = label_num
        self.epoch_tr_loss, self.epoch_vl_loss = [], []
        self.epoch_tr_acc, self.epoch_vl_acc = [], []
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.double = double
        self.class_num = class_num
        self.is_al = is_al

        if self.double:
            assert type(self.class_num) == int  # this is for double label

        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
            print("GPU is available")
        else:
            self.device = torch.device("cpu")
            print("GPU not available, CPU used")

        self.best = 0

    def run(self, epochs):

        self.valid_acc_min = -999
        for e in range(epochs):

            total_loss = []
            total_emb_loss = []
            losses = {
                "emb bridge": [],
                "emb associated": [],
                "layer1 bridge": [],
                "layer1 associated": [],
                "layer2 bridge": [],
                "layer2 associated": []
            }
            total_acc = 0.0
            total_count = 0

            for inputs, masks, labels in self.train_loader:

                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                self.model.train()
                self.opt.zero_grad()

                if self.is_al:

                    emb_loss, layers_loss = self.model(
                        inputs, labels, src_key_padding_mask=masks
                    )

                    total_emb_loss.append(emb_loss.item())
                    loss = emb_loss
                    for l in layers_loss:
                        loss += l

                    losses["emb bridge"].append(self.model.embedding.loss_b)
                    losses["emb associated"].append(
                        self.model.embedding.loss_d)
                    losses["layer1 bridge"].append(self.model.layer_1.loss_b)
                    losses["layer1 associated"].append(
                        self.model.layer_1.loss_d)
                    losses["layer2 bridge"].append(self.model.layer_2.loss_b)
                    losses["layer2 associated"].append(
                        self.model.layer_2.loss_d)

                else:

                    outputs = self.model(inputs, src_key_padding_mask=masks)
                    loss = self.cri(
                        outputs, torch.argmax(labels.long(), dim=1))

                loss.backward()
                total_loss.append(loss.item())

                self.opt.step()
                torch.cuda.empty_cache()

                # calculate the loss and perform backprop
                with torch.no_grad():

                    self.model.eval()

                    if self.is_al:

                        left = self.model.embedding.f(inputs)
                        left = self.model.layer_1.f(
                            left, None, masks)
                        # NOTE: deprecated codes. for nn.Module list
                        # for l in self.model.layers:
                        #     left = l.f(
                        #         left, src_key_padding_mask=masks)
                        left = self.model.layer_2.f(
                            left, None, masks)
                        # left = self.model.layer_3.f(
                        #     left, None, masks)
                        # left = self.model.layer_4.f(
                        #     left, None, masks)
                        # left = self.model.layer_5.f(
                        #     left, None, masks)
                        # left = self.model.layer_6.f(
                        #     left, None, masks)

                        # mean pooling
                        left = left.sum(dim=1)
                        src_len = (masks == 0).sum(dim=1)
                        src_len = torch.stack((src_len,) * left.size(1), dim=1)
                        left = left / src_len

                        right = self.model.layer_2.bx(left)

                        # NOTE: deprecated codes. for nn.Module list
                        # right = self.model.layers[-1].bx(left)
                        # for l in self.model.layers:
                        #     right = l.dy(right)
                        # right = self.model.layer_6.dy(right)
                        # right = self.model.layer_5.dy(right)
                        # right = self.model.layer_4.dy(right)
                        # right = self.model.layer_3.dy(right)
                        right = self.model.layer_2.dy(right)
                        right = self.model.layer_1.dy(right)

                        if self.label_num == 2:
                            predicted_label = torch.round(
                                self.model.embedding.dy(right).squeeze())
                            total_acc += (predicted_label ==
                                          labels.to(torch.float)).sum().item()
                        else:
                            if self.double is False:
                                predicted_label = self.model.embedding.dy(
                                    right)
                            else:
                                out = self.model.embedding.dy(right)
                                predicted_label = out[:, :self.class_num] + \
                                    out[:, self.class_num:]
                                labels = labels[:, :self.class_num] + \
                                    labels[:, self.class_num:]

                    else:
                        predicted_label = outputs

                    total_acc += (predicted_label.argmax(-1) ==
                                  labels.to(torch.float).argmax(-1)).sum().item()
                    total_count += labels.size(0)

            val_acc = 0.0
            val_count = 0

            for inputs, masks, labels in self.valid_loader:

                with torch.no_grad():

                    inputs = inputs.to(self.device)
                    masks = masks.to(self.device)
                    labels = labels.to(self.device)

                    self.model.eval()

                    if self.is_al:

                        left = self.model.embedding.f(inputs)
                        left = self.model.layer_1.f(
                            left, None, masks)
                        # NOTE: deprecated codes. for nn.Module list
                        # for l in self.model.layers:
                        #     left = l.f(
                        #         left, src_key_padding_mask=masks)
                        left = self.model.layer_2.f(
                            left, None, masks)
                        # left = self.model.layer_3.f(
                        #     left, None, masks)
                        # left = self.model.layer_4.f(
                        #     left, None, masks)
                        # left = self.model.layer_5.f(
                        #     left, None, masks)
                        # left = self.model.layer_6.f(
                        #     left, None, masks)

                        # mean pooling
                        left = left.sum(dim=1)
                        src_len = (masks == 0).sum(dim=1)
                        src_len = torch.stack((src_len,) * left.size(1), dim=1)
                        left = left / src_len

                        right = self.model.layer_2.bx(left)

                        # NOTE: deprecated codes. for nn.Module list
                        # right = self.model.layers[-1].bx(left)
                        # for l in self.model.layers:
                        #     right = l.dy(right)
                        # right = self.model.layer_6.dy(right)
                        # right = self.model.layer_5.dy(right)
                        # right = self.model.layer_4.dy(right)
                        # right = self.model.layer_3.dy(right)
                        right = self.model.layer_2.dy(right)
                        right = self.model.layer_1.dy(right)

                        if self.label_num == 2:
                            predicted_label = self.model.embedding.dy(
                                right).squeeze()

                        else:
                            if self.double is False:
                                predicted_label = self.model.embedding.dy(
                                    right)
                            else:
                                out = self.model.embedding.dy(right)
                                predicted_label = out[:, :self.class_num] + \
                                    out[:, self.class_num:]
                                labels = labels[:, :self.class_num] + \
                                    labels[:, self.class_num:]

                    else:

                        predicted_label = self.model(
                            inputs, src_key_padding_mask=masks
                        )

                    val_acc += (predicted_label.argmax(-1) ==
                                labels.to(torch.float).argmax(-1)).sum().item()
                    val_count += labels.size(0)

            epoch_train_loss = [np.mean(total_emb_loss), np.mean(
                total_loss), ]
            epoch_train_acc = total_acc/total_count
            epoch_val_acc = val_acc/val_count

            for k in list(losses.keys()):
                losses[k] = np.mean(losses[k])

            wandb.log(losses)

            print(f'Epoch {e+1}')
            if self.is_al:
                print(
                    f'train_loss : emb loss {epoch_train_loss[0]}, total loss {epoch_train_loss[1]}')
                print(
                    f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
            else:
                print(
                    f'training loss {np.mean(total_loss)}'
                )
                print(
                    f'train acc: {epoch_train_acc*100} val acc: {epoch_val_acc*100}'
                )
            if epoch_val_acc >= self.valid_acc_min:
                torch.save(self.model.state_dict(), f'{self.save_dir}')
                print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    self.valid_acc_min, epoch_val_acc))
                self.valid_acc_min = epoch_val_acc
                self.best = e+1
            print(25*'==')
        final_dp = self.save_dir[:-3] + 'last.pth'
        torch.save(self.model.state_dict(), f'{final_dp}')
        print('best val acc', self.valid_acc_min)

    def eval(self):

        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)

        test_acc = 0
        test_count = 0

        for inputs, masks, labels in self.test_loader:

            with torch.no_grad():

                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                if self.is_al:

                    left = self.model.embedding.f(inputs)
                    left = self.model.layer_1.f(
                        left, None, masks)
                    # NOTE: deprecated codes. for nn.Module list
                    # for l in self.model.layers:
                    #     left = l.f(
                    #         left, src_key_padding_mask=masks)
                    left = self.model.layer_2.f(
                        left, None, masks)
                    # left = self.model.layer_3.f(
                    #     left, None, masks)
                    # left = self.model.layer_4.f(
                    #     left, None, masks)
                    # left = self.model.layer_5.f(
                    #     left, None, masks)
                    # left = self.model.layer_6.f(
                    #     left, None, masks)

                    # mean pooling
                    left = left.sum(dim=1)
                    src_len = (masks == 0).sum(dim=1)
                    src_len = torch.stack((src_len,) * left.size(1), dim=1)
                    left = left / src_len

                    right = self.model.layer_2.bx(left)

                    # NOTE: deprecated codes. for nn.Module list
                    # right = self.model.layers[-1].bx(left)
                    # for l in self.model.layers:
                    #     right = l.dy(right)
                    # right = self.model.layer_6.dy(right)
                    # right = self.model.layer_5.dy(right)
                    # right = self.model.layer_4.dy(right)
                    # right = self.model.layer_3.dy(right)
                    right = self.model.layer_2.dy(right)
                    right = self.model.layer_1.dy(right)

                    if self.label_num == 2:
                        predicted_label = model.embedding.dy(right).squeeze()
                    else:
                        if self.double is False:
                            predicted_label = self.model.embedding.dy(right)
                        else:
                            out = self.model.embedding.dy(right)
                            predicted_label = out[:, :self.class_num] + \
                                out[:, self.class_num:]
                            labels = labels[:, :self.class_num] + \
                                labels[:, self.class_num:]

                else:

                    predicted_label = self.model(
                        inputs, src_key_padding_mask=masks
                    )

                test_acc += (predicted_label.argmax(-1) ==
                             labels.to(torch.float).argmax(-1)).sum().item()

                test_count += labels.size(0)

        print(f'Test acc:{test_acc/test_count*100}')

    def pred(self):
        '''for sst2'''

        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)

        pred_file = {
            'index': [],
            'prediction': []
        }

        for idx, (inputs, masks) in enumerate(self.test_loader):

            with torch.no_grad():

                inputs = inputs.to(self.device)
                masks = masks.to(self.device)

                if self.is_al:

                    left = self.model.embedding.f(inputs)
                    left = self.model.layer_1.f(
                        left, None, masks)
                    # NOTE: deprecated codes. for nn.Module list
                    # for l in self.model.layers:
                    #     left = l.f(
                    #         left, src_key_padding_mask=masks)
                    left = self.model.layer_2.f(
                        left, None, masks)
                    # left = self.model.layer_3.f(
                    #     left, None, masks)
                    # left = self.model.layer_4.f(
                    #     left, None, masks)
                    # left = self.model.layer_5.f(
                    #     left, None, masks)
                    # left = self.model.layer_6.f(
                    #     left, None, masks)

                    # mean pooling
                    left = left.sum(dim=1)
                    src_len = (masks == 0).sum(dim=1)
                    src_len = torch.stack((src_len,) * left.size(1), dim=1)
                    left = left / src_len

                    right = self.model.layer_2.bx(left)

                    # NOTE: deprecated codes. for nn.Module list
                    # right = self.model.layers[-1].bx(left)
                    # for l in self.model.layers:
                    #     right = l.dy(right)
                    # right = self.model.layer_6.dy(right)
                    # right = self.model.layer_5.dy(right)
                    # right = self.model.layer_4.dy(right)
                    # right = self.model.layer_3.dy(right)
                    right = self.model.layer_2.dy(right)
                    right = self.model.layer_1.dy(right)

                    predicted_label = model.embedding.dy(
                        right)

                else:

                    predicted_label = self.model(
                        inputs, src_key_padding_mask=masks
                    )

                pred_file['index'].append(idx)
                pred_file['prediction'].append(
                    predicted_label.argmax(1).item())

        output = pd.DataFrame(pred_file)
        output.to_csv('data/sst2/SST-2.tsv', sep='\t', index=False)

    def short_cut_emb(self):

        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)

        test_acc = 0
        test_count = 0

        for inputs, masks, labels in self.test_loader:

            with torch.no_grad():

                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                right = model.short_cut_emb(inputs)

                predicted_label = right.squeeze()
                test_acc += (predicted_label.argmax(-1) ==
                             labels.to(torch.float).argmax(-1)).sum().item()

                test_count += labels.size(0)

        print('Short cut emb Test acc', test_acc/test_count)

    def short_cut_l1(self):

        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))

        test_acc = 0
        test_count = 0

        for inputs, masks, labels in self.test_loader:

            with torch.no_grad():

                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                right = self.model.short_cut_l1(inputs, masks)

                predicted_label = right.squeeze()
                test_acc += (predicted_label.argmax(-1) ==
                             labels.to(torch.float).argmax(-1)).sum().item()

                test_count += labels.size(0)

        print('Short cut layer1 Test acc', test_acc/test_count)


class ALTrainer:

    def __init__(
        self, model, lr, train_loader, valid_loader, test_loader, save_dir, label_num=None, double=False, class_num=None
    ):

        self.mi = MI_Vis()
        self.sample = []

        self.model = model
        project_name = save_dir.replace('/', '-')
        wandb.init(project=project_name, entity='al-train')
        config = wandb.config
        wandb.watch(self.model)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.opt = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.label_num = label_num
        self.epoch_tr_loss, self.epoch_vl_loss = [], []
        self.epoch_tr_acc, self.epoch_vl_acc = [], []
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.double = double
        self.class_num = class_num

        if self.double:
            assert type(self.class_num) == int  # this is for double label

        is_cuda = torch.cuda.is_available()

        if is_cuda:
            self.device = torch.device("cuda")
            print("GPU is available")
        else:
            self.device = torch.device("cpu")
            print("GPU not available, CPU used")

        self.ckpt_epoch = 0
        self.pat = 0

    def run(self, epochs):

        self.valid_acc_min = -999
        for epoch in range(epochs):
            train_losses = []

            total_emb_loss = []
            total_l1_loss = []
            total_l2_loss = []

            total_acc = 0.0
            total_count = 0
            self.model.embedding.train()
            self.model.layer_1.train()
            self.model.layer_2.train()

            # initialize hidden state
            for inputs, labels in self.train_loader:

                self.model.train()
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.opt.zero_grad()

                emb_loss, l1_loss, l2_loss = self.model(inputs, labels)

                wandb.log({"emb loss": emb_loss.item()})
                wandb.log({"lstm1 loss": l1_loss.item()})
                wandb.log({"lstm2 loss": l2_loss.item()})
                wandb.log({"emb bridge loss": self.model.embedding.loss_b})
                wandb.log({"emb decode loss": self.model.embedding.loss_d})
                wandb.log({"lstm1 bridge loss": self.model.layer_1.loss_b})
                wandb.log({"lstm1 decode loss": self.model.layer_1.loss_d})
                wandb.log({"lstm2 bridge loss": self.model.layer_2.loss_b})
                wandb.log({"lstm2 decode loss": self.model.layer_2.loss_d})

                loss = emb_loss + l1_loss + l2_loss
                wandb.log({"total loss": loss.item()})
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.model.layer_1.parameters(), 5, error_if_nonfinite=True)
                nn.utils.clip_grad_norm_(
                    self.model.layer_2.parameters(), 5, error_if_nonfinite=True)

                total_emb_loss.append(emb_loss.item())
                total_l1_loss.append(l1_loss.item())
                total_l2_loss.append(l2_loss.item())

                self.opt.step()

                torch.cuda.empty_cache()

                # calculate the loss and perform backprop
                with torch.no_grad():

                    self.model.eval()

                    left = self.model.embedding.f(inputs)
                    # left = self.model.ln(left)

                    output, hidden = self.model.layer_1.f(left)
                    # output = self.model.ln2(output)

                    left, (output, c) = self.model.layer_2.f(output, hidden)
                    left = left[:, -1, :]

                    right = self.model.layer_2.bx(left)
                    right = self.model.layer_2.dy(right)
                    right = self.model.layer_1.dy(right)
                    if self.label_num == 2:
                        predicted_label = torch.round(
                            self.model.embedding.dy(right).squeeze())
                        total_acc += (predicted_label ==
                                      labels.to(torch.float)).sum().item()
                    else:
                        if self.double is False:
                            predicted_label = self.model.embedding.dy(right)
                        else:
                            out = self.model.embedding.dy(right)
                            predicted_label = out[:, :self.class_num] + \
                                out[:, self.class_num:]
                            labels = labels[:, :self.class_num] + \
                                labels[:, self.class_num:]
                        # print(predicted_label)
                        # print(labels)
                        # raise Exception
                        total_acc += (predicted_label.argmax(-1) ==
                                      labels.to(torch.float).argmax(-1)).sum().item()

                    total_count += labels.size(0)

            val_losses = []
            val_acc = 0.0
            val_count = 0

            self.model.embedding.eval()
            self.model.layer_1.eval()
            self.model.layer_2.eval()

            for inputs, labels in self.valid_loader:

                with torch.no_grad():

                    self.model.embedding.eval()
                    self.model.layer_1.eval()
                    self.model.layer_2.eval()

                    inputs, labels = inputs.to(
                        self.device), labels.to(self.device)

                    left = self.model.embedding.f(inputs)
                    output, hidden = self.model.layer_1.f(left)

                    left, (output, c) = self.model.layer_2.f(output, hidden)
                    left = left[:, -1, :]

                    right = self.model.layer_2.bx(left)
                    right = self.model.layer_2.dy(right)
                    right = self.model.layer_1.dy(right)
                    if self.label_num == 2:
                        predicted_label = torch.round(
                            self.model.embedding.dy(right).squeeze())
                        val_acc += (predicted_label ==
                                    labels.to(torch.float)).sum().item()
                    else:
                        if self.double is False:
                            predicted_label = self.model.embedding.dy(right)
                        else:
                            out = self.model.embedding.dy(right)
                            predicted_label = out[:, :self.class_num] + \
                                out[:, self.class_num:]
                            labels = labels[:, :self.class_num] + \
                                labels[:, self.class_num:]

                        val_acc += (predicted_label.argmax(-1) ==
                                    labels.to(torch.float).argmax(-1)).sum().item()

                    val_count += labels.size(0)

            epoch_train_loss = [np.mean(total_emb_loss), np.mean(
                total_l1_loss), np.mean(total_l2_loss)]
            epoch_train_acc = total_acc/total_count
            epoch_val_acc = val_acc/val_count

            print(f'Epoch {epoch+1}')
            print(
                f'train_loss : emb loss {epoch_train_loss[0]}, layer1 loss {epoch_train_loss[1]}, layer2 loss {epoch_train_loss[2]}')
            print(
                f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
            wandb.log({"train acc": epoch_train_acc})
            wandb.log({"valid acc": epoch_val_acc})

            if epoch_val_acc >= self.valid_acc_min:
                self.pat = 0
                self.ckpt_epoch = epoch+1
                torch.save(self.model.state_dict(), f'{self.save_dir}')
                print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    self.valid_acc_min, epoch_val_acc))
                self.valid_acc_min = epoch_val_acc
            print(25*'==')
        final_dp = self.save_dir[:-3] + 'last.pth'
        torch.save(self.model.state_dict(), f'{final_dp}')
        print('best val acc', self.valid_acc_min)
        print('best checkpoint at', self.ckpt_epoch, 'epoch')

    def eval(self):

        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)

        test_acc = 0
        test_count = 0

        for inputs, labels in self.test_loader:

            with torch.no_grad():

                model.embedding.eval()
                model.layer_1.eval()
                model.layer_2.eval()

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                left = model.embedding.f(inputs)
                output, hidden = model.layer_1.f(left)
                left, (output, c) = model.layer_2.f(output, hidden)
                left = left[:, -1, :]
                # left = left.reshape(left.size(1), -1)
                right = model.layer_2.bx(left)
                right = model.layer_2.dy(right)
                right = model.layer_1.dy(right)
                if self.label_num == 2:
                    predicted_label = torch.round(
                        model.embedding.dy(right).squeeze())
                    test_acc += (predicted_label ==
                                 labels.to(torch.float)).sum().item()
                else:
                    if self.double is False:
                        predicted_label = self.model.embedding.dy(right)
                    else:
                        out = self.model.embedding.dy(right)
                        predicted_label = out[:, :self.class_num] + \
                            out[:, self.class_num:]
                        labels = labels[:, :self.class_num] + \
                            labels[:, self.class_num:]

                    test_acc += (predicted_label.argmax(-1) ==
                                 labels.to(torch.float).argmax(-1)).sum().item()

                test_count += labels.size(0)
        x = test_acc/test_count
        wandb.log({"test acc": x})
        print('Test acc', test_acc/test_count)
        # self.tsne_()

    def short_cut_emb(self):

        self.model.eval()
        # self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)

        test_acc = 0
        test_count = 0

        for inputs, labels in self.test_loader:

            with torch.no_grad():

                model.embedding.eval()
                model.layer_1.eval()
                model.layer_2.eval()

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                right = model.short_cut_emb(inputs)

                if self.label_num == 2:
                    predicted_label = torch.round(
                        model.embedding.dy(right).squeeze())
                    test_acc += (predicted_label ==
                                 labels.to(torch.float)).sum().item()
                else:
                    predicted_label = right.squeeze()
                    test_acc += (predicted_label.argmax(-1) ==
                                 labels.to(torch.float).argmax(-1)).sum().item()

                test_count += labels.size(0)

        print('Short cut emb Test acc', test_acc/test_count)

    def short_cut_l1(self):

        self.model.eval()
        # self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)

        test_acc = 0
        test_count = 0

        for inputs, labels in self.test_loader:

            with torch.no_grad():

                model.embedding.eval()
                model.layer_1.eval()
                model.layer_2.eval()

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                right = model.short_cut_lstm(inputs)

                if self.label_num == 2:
                    predicted_label = torch.round(
                        model.embedding.dy(right).squeeze())
                    test_acc += (predicted_label ==
                                 labels.to(torch.float)).sum().item()
                else:
                    predicted_label = right.squeeze()
                    test_acc += (predicted_label.argmax(-1) ==
                                 labels.to(torch.float).argmax(-1)).sum().item()

                test_count += labels.size(0)

        print('Short cut lstm Test acc', test_acc/test_count)

    def tsne_(self):
        print('working on tsne')
        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        self.model = self.model.to(self.device)
        all_feats = []
        all_labels = []
        class_num = 0
        for inputs, labels in self.test_loader:

            with torch.no_grad():

                self.model.embedding.eval()
                self.model.layer_1.eval()
                self.model.layer_2.eval()

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                left = self.model.embedding.f(inputs)
                output, hidden = self.model.layer_1.f(left)
                left, (output, c) = self.model.layer_2.f(output, hidden)
                left = left[:, -1, :]
                # left = left.reshape(left.size(1), -1)
                right = self.model.layer_2.bx(left)
                right = self.model.layer_2.dy(right)
                right = right.cpu()
                labels = labels.cpu()
                # left = left.reshape(left.size(1), -1)
                for i in range(right.size(0)):
                    all_feats.append(right[i, :].numpy())
                    all_labels.append(labels[i, :].argmax(-1).item())

        tsne_plot_dir = self.save_dir[:-2]+'al2.tsne.png'
        tsne(all_feats, all_labels, class_num, tsne_plot_dir)
        # print('tsne saved in ', tsne_plot_dir)


class Trainer:

    def __init__(
        self, model, lr, train_loader, valid_loader, test_loader, save_dir, label_num=None, loss_w=None, only_pred=None
    ):

        self.model = model

        project_name = save_dir.replace('/', '-normal-')
        wandb.init(project=project_name, entity='al-train')
        config = wandb.config
        wandb.watch(self.model)
        # self.opt = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.label_num = label_num
        self.epoch_tr_loss, self.epoch_vl_loss = [], []
        self.epoch_tr_acc, self.epoch_vl_acc = [], []
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_dir = save_dir
        if loss_w is not None:
            self.cri = nn.CrossEntropyLoss(weight=loss_w.cuda())
        else:
            self.cri = nn.CrossEntropyLoss()
        self.clip = 1
        is_cuda = torch.cuda.is_available()

        if is_cuda:
            self.device = torch.device("cuda")
            print("GPU is available")
        else:
            self.device = torch.device("cpu")
            print("GPU not available, CPU used")

        self.pat = 0
        self.ckpt_epoch = 0

    def run(self, epochs):

        self.valid_acc_min = -999
        # train for some number of epochs
        epoch_tr_loss, epoch_vl_loss = [], []
        epoch_tr_acc, epoch_vl_acc = [], []

        for epoch in range(epochs):
            train_losses = []
            train_acc = 0.0
            self.model.train()
            train_count = 0
            for inputs, labels in self.train_loader:

                # labels = torch.argmax(labels.long(), dim=1)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.opt.zero_grad()
                output, h = self.model(inputs)
                loss = self.cri(output, labels)
                loss.backward()
                train_losses.append(loss.item())
                # accuracy = acc(output,labels)
                train_acc += (output.argmax(-1) == labels.float()).sum().item()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip, error_if_nonfinite=True)
                self.opt.step()
                train_count += labels.size(0)

            val_losses = []
            val_acc = 0.0
            val_count = 0
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.valid_loader:

                    inputs, labels = inputs.to(
                        self.device), labels.to(self.device)
                    output, val_h = self.model(inputs)
                    val_loss = self.cri(output, labels)
                    val_losses.append(val_loss.item())
                    # accuracy = acc(output,labels)
                    val_acc += (output.argmax(-1) ==
                                labels.float()).sum().item()
                    val_count += labels.size(0)

            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            epoch_train_acc = train_acc/train_count
            epoch_val_acc = val_acc/val_count
            epoch_tr_loss.append(epoch_train_loss)
            epoch_vl_loss.append(epoch_val_loss)
            epoch_tr_acc.append(epoch_train_acc)
            epoch_vl_acc.append(epoch_val_acc)
            print(f'Epoch {epoch+1}')
            print(
                f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
            print(
                f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')

            wandb.log({"train acc": epoch_train_acc*100})
            wandb.log({"valid acc": epoch_val_acc*100})
            if epoch_val_acc >= self.valid_acc_min:
                self.pat = 0
                self.ckpt_epoch = epoch+1
                torch.save(self.model.state_dict(), f'{self.save_dir}')
                print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    self.valid_acc_min, epoch_val_acc))
                self.valid_acc_min = epoch_val_acc
            print(25*'==')
        print('best valid acc', self.valid_acc_min)
        print('best checkpoint at', self.ckpt_epoch)

    def eval(self):
        test_losses = []  # track loss
        num_correct = 0
        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        test_count = 0
        # iterate over test data
        for inputs, labels in self.test_loader:

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            output, test_h = self.model(inputs)
            # calculate loss
            test_loss = self.cri(output, labels)
            test_losses.append(test_loss.item())
            pred = output.argmax(-1)

            # correct_tensor = pred.eq(labels.float().view_as(pred))
            # correct = np.squeeze(correct_tensor.cpu().numpy())
            num_correct += (pred == labels.float()).sum().item()
            test_count += labels.size(0)

        print("Test loss: {:.3f}".format(np.mean(test_losses)))

        test_acc = num_correct/test_count
        print("Test accuracy: {:.3f}".format(test_acc))
        # self.tsne_()

    def pred(self):
        test_losses = []  # track loss
        num_correct = 0
        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        test_count = 0
        pred_list = []
        # iterate over test data
        for inputs, labels in self.test_loader:

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            output, test_h = self.model(inputs)
            # calculate loss
            # test_loss = self.cri(output, labels)
            # test_losses.append(test_loss.item())
            pred = output.argmax(-1)
            pred_list = pred_list + pred.tolist()
        tsv_name = self.save_dir[:-3]+'.tsv'
        df = {'index': [i for i in range(
            len(pred_list))], 'prediction': pred_list}
        df = pd.DataFrame(df)
        df.to_csv(tsv_name, sep='\t', index=False)

    def tsne_(self):

        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)
        all_feats = []
        all_labels = []
        class_num = 0
        for inputs, labels in self.test_loader:

            with torch.no_grad():
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                embeds = self.model.embedding(inputs)
                lstm_out, hidden = self.model.lstm(embeds)
                right = lstm_out[:, -1, :]
                right = right.cpu()
                labels = labels.cpu()
                for i in range(right.size(0)):
                    all_feats.append(right[i, :].numpy())
                    all_labels.append(labels[i, :].argmax(-1).item())

        tsne_plot_dir = self.save_dir[:-2]+'lstm.tsne.png'
        tsne(all_feats, all_labels, class_num, tsne_plot_dir)
