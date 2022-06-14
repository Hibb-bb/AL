# -*- coding: utf-8 -*-
import random
import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.io import read_image

from datasets import load_dataset

from ..utils import (Padding, PadTransformer, convert2id, create_vocab,
                     data_preprocessing, multi_class_process)


class AGNews(object):

    def __init__(self) -> None:
        super(AGNews, self).__init__()

    def load(self, args):

        news_train = load_dataset('ag_news', split='train')
        new_test = load_dataset('ag_news', split='test')

        class_num = args.class_num

        train_text = [b['text'] for b in news_train]
        train_label = multi_class_process(
            [b['label'] for b in news_train], class_num)

        test_text = [b['text'] for b in new_test]
        test_label = multi_class_process(
            [b['label'] for b in new_test], class_num)

        clean_train = [data_preprocessing(t) for t in train_text]
        clean_test = [data_preprocessing(t) for t in test_text]

        vocab = create_vocab(clean_train)

        clean_train_id = convert2id(clean_train, vocab)
        clean_test_id = convert2id(clean_test, vocab)

        max_len = max([len(s) for s in clean_train_id])
        print('max seq length', max_len)

        train_features = Padding(clean_train_id, max_len)
        test_features = Padding(clean_test_id, max_len)

        X_train, X_valid, y_train, y_valid = train_test_split(
            train_features, train_label, test_size=0.2, random_state=1)
        X_test, y_test = test_features, test_label

        print('dataset information:')
        print('=====================')
        print('train size', len(X_train))
        print('valid size', len(X_valid))
        print('test size', len(test_features))
        print('=====================')

        train_data = TensorDataset(
            torch.from_numpy(X_train), torch.stack(y_train))
        test_data = TensorDataset(
            torch.from_numpy(X_test), torch.stack(y_test))
        valid_data = TensorDataset(
            torch.from_numpy(X_valid), torch.stack(y_valid))

        batch_size = args.batch_size

        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(
            test_data, shuffle=False, batch_size=batch_size)
        valid_loader = DataLoader(
            valid_data, shuffle=False, batch_size=batch_size)

        return train_loader, valid_loader, test_loader, vocab

    def load_with_masks(self, args):

        news_train = load_dataset('ag_news', split='train')
        news_test = load_dataset('ag_news', split='test')

        # TODO: columns
        train_text = [b['text'] for b in news_train]
        train_label = multi_class_process(
            [b['label'] for b in news_train], args.class_num
        )
        test_text = [b['text'] for b in news_test]
        test_label = multi_class_process(
            [b['label'] for b in news_test], args.class_num
        )

        clean_train = [data_preprocessing(t, True) for t in train_text]
        clean_test = [data_preprocessing(t, True) for t in test_text]

        vocab = create_vocab(clean_train)

        clean_train_id = convert2id(clean_train, vocab)
        clean_test_id = convert2id(clean_test, vocab)

        max_len = max([len(s) for s in clean_train_id])
        args.max_len = min(max_len, args.max_len)
        print('max seq length', max_len)

        train_features, train_mask = PadTransformer(
            clean_train_id, args.max_len)
        test_features, test_mask = PadTransformer(clean_test_id, args.max_len)

        X_train, X_valid, mask_train, mask_valid, y_train, y_valid = train_test_split(
            train_features, train_mask, train_label, test_size=0.2, random_state=1)
        X_test, mask_test, y_test = test_features, test_mask, test_label

        print('dataset information:')
        print('=====================')
        print('train size', len(X_train))
        print('valid size', len(X_valid))
        print('test size', len(test_features))
        print('=====================')

        train_data = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(mask_train),
            torch.stack(y_train)
        )
        test_data = TensorDataset(
            torch.from_numpy(X_test),
            torch.from_numpy(mask_test),
            torch.stack(y_test)
        )
        valid_data = TensorDataset(
            torch.from_numpy(X_valid),
            torch.from_numpy(mask_valid),
            torch.stack(y_valid)
        )

        batch_size = args.batch_size

        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(
            test_data, shuffle=False, batch_size=batch_size)
        valid_loader = DataLoader(
            valid_data, shuffle=False, batch_size=batch_size)

        return train_loader, valid_loader, test_loader, vocab


class DBpedia(object):

    def __init__(self) -> None:
        super(DBpedia, self).__init__()

    def load(self, args):

        news_train = load_dataset('dbpedia_14', split='train')
        new_test = load_dataset('dbpedia_14', split='test')

        class_num = args.class_num

        train_text = [b['content'] for b in news_train]
        train_label = multi_class_process(
            [b['label'] for b in news_train], class_num)
        test_text = [b['content'] for b in new_test]
        test_label = multi_class_process(
            [b['label'] for b in new_test], class_num)

        if args.random_label:
            train_label = multi_class_process(
                [random.randint(0, 19) for _ in range(len(train_text))], class_num)
            args.save_dir = args.save_dir[:-2]+'.rand.pt'
            print('This is a random label test')

        clean_train = [data_preprocessing(t, True) for t in train_text]

        clean_test = [data_preprocessing(t, True) for t in test_text]

        vocab = create_vocab(clean_train)

        clean_train_id = convert2id(clean_train, vocab)
        clean_test_id = convert2id(clean_test, vocab)

        max_len = max([len(s) for s in clean_train_id])
        print('max seq length', max_len)
        max_len = args.max_len

        train_features = Padding(clean_train_id, max_len)
        test_features = Padding(clean_test_id, max_len)

        X_train, X_valid, y_train, y_valid = train_test_split(
            train_features, train_label, test_size=0.2, random_state=1)

        print('dataset information:')
        print('=====================')
        print('train size', len(X_train))
        print('valid size', len(X_valid))
        print('test size', len(test_features))
        print('=====================')
        X_test, y_test = test_features, test_label

        train_data = TensorDataset(torch.from_numpy(
            X_train), torch.stack(y_train, dim=0))
        test_data = TensorDataset(torch.from_numpy(
            X_test), torch.stack(y_test, dim=0))
        valid_data = TensorDataset(torch.from_numpy(
            X_valid), torch.stack(y_valid, dim=0))

        batch_size = args.batch_size

        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(
            test_data, shuffle=False, batch_size=batch_size)
        valid_loader = DataLoader(
            valid_data, shuffle=False, batch_size=batch_size)

        return train_loader, valid_loader, test_loader, vocab

    def load_with_masks(self, args):

        news_train = load_dataset('dbpedia_14', split='train')
        news_test = load_dataset('dbpedia_14', split='test')

        # TODO: columns
        train_text = [b['content'] for b in news_train]
        train_label = multi_class_process(
            [b['label'] for b in news_train], args.class_num
        )
        test_text = [b['content'] for b in news_test]
        test_label = multi_class_process(
            [b['label'] for b in news_test], args.class_num
        )

        clean_train = [data_preprocessing(t, True) for t in train_text]
        clean_test = [data_preprocessing(t, True) for t in test_text]

        vocab = create_vocab(clean_train)

        clean_train_id = convert2id(clean_train, vocab)
        clean_test_id = convert2id(clean_test, vocab)

        max_len = max([len(s) for s in clean_train_id])
        args.max_len = min(max_len, args.max_len)
        print('max seq length', max_len)

        train_features, train_mask = PadTransformer(
            clean_train_id, args.max_len)
        test_features, test_mask = PadTransformer(clean_test_id, args.max_len)

        X_train, X_valid, mask_train, mask_valid, y_train, y_valid = train_test_split(
            train_features, train_mask, train_label, test_size=0.2, random_state=1)
        X_test, mask_test, y_test = test_features, test_mask, test_label

        print('dataset information:')
        print('=====================')
        print('train size', len(X_train))
        print('valid size', len(X_valid))
        print('test size', len(test_features))
        print('=====================')

        train_data = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(mask_train),
            torch.stack(y_train)
        )
        test_data = TensorDataset(
            torch.from_numpy(X_test),
            torch.from_numpy(mask_test),
            torch.stack(y_test)
        )
        valid_data = TensorDataset(
            torch.from_numpy(X_valid),
            torch.from_numpy(mask_valid),
            torch.stack(y_valid)
        )

        batch_size = args.batch_size

        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(
            test_data, shuffle=False, batch_size=batch_size)
        valid_loader = DataLoader(
            valid_data, shuffle=False, batch_size=batch_size)

        return train_loader, valid_loader, test_loader, vocab


class IMDb(object):

    def __init__(self) -> None:
        super(IMDb, self).__init__()

    def load(self, args):

        train_df = pd.read_csv('data/imdb/train.csv')
        test_df = pd.read_csv('data/imdb/test.csv')

        # TODO: columns
        train_text = train_df['text'].tolist()
        train_label = multi_class_process(train_df['label'].tolist(), 2)

        test_text = test_df['text'].tolist()
        test_label = multi_class_process(test_df['label'].tolist(), 2)

        clean_train = [data_preprocessing(t, True) for t in train_text]
        clean_test = [data_preprocessing(t, True) for t in test_text]

        vocab = create_vocab(clean_train)

        clean_train_id = convert2id(clean_train, vocab)
        clean_test_id = convert2id(clean_test, vocab)

        max_len = max([len(s) for s in clean_train_id])
        print(f'max seq length: {max_len}')
        print(f'len limit: {args.max_len}')

        train_features = Padding(clean_train_id, args.max_len)
        test_features = Padding(clean_test_id, args.max_len)

        X_train, X_valid, y_train, y_valid = train_test_split(
            train_features, train_label, test_size=0.2, random_state=1
        )
        X_test, y_test = test_features, test_label

        # create tensor dataset
        train_data = TensorDataset(torch.from_numpy(
            X_train), torch.stack(y_train, dim=0))
        test_data = TensorDataset(torch.from_numpy(
            X_test), torch.stack(y_test, dim=0))
        valid_data = TensorDataset(torch.from_numpy(
            X_valid), torch.stack(y_valid, dim=0))

        print('dataset information:')
        print('=====================')
        print('train size', len(X_train))
        print('valid size', len(X_valid))
        print('test size', len(X_test))
        print('=====================')

        batch_size = args.batch_size

        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(
            test_data, shuffle=False, batch_size=batch_size)
        valid_loader = DataLoader(
            valid_data, shuffle=False, batch_size=batch_size)

        return train_loader, valid_loader, test_loader, vocab

    def load_with_masks(self, args):

        train_df = pd.read_csv('data/imdb/train.csv')
        test_df = pd.read_csv('data/imdb/test.csv')

        # TODO: columns
        train_text = train_df['text'].tolist()
        train_label = multi_class_process(train_df['label'].tolist(), 2)

        test_text = test_df['text'].tolist()
        test_label = multi_class_process(test_df['label'].tolist(), 2)

        clean_train = [data_preprocessing(t, True) for t in train_text]
        clean_test = [data_preprocessing(t, True) for t in test_text]

        vocab = create_vocab(clean_train)

        clean_train_id = convert2id(clean_train, vocab)
        clean_test_id = convert2id(clean_test, vocab)

        max_len = max([len(s) for s in clean_train_id])
        print(f'max seq length: {max_len}')
        print(f'len limit: {args.max_len}')

        train_features, train_mask = PadTransformer(
            clean_train_id, args.max_len)
        test_features, test_mask = PadTransformer(clean_test_id, args.max_len)

        X_train, X_valid, mask_train, mask_valid, y_train, y_valid = train_test_split(
            train_features, train_mask, train_label, test_size=0.2, random_state=1
        )
        X_test, mask_test, y_test = test_features, test_mask, test_label

        print('dataset information:')
        print('=====================')
        print('train size', len(X_train))
        print('valid size', len(X_valid))
        print('test size', len(test_features))
        print('=====================')

        train_data = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(mask_train),
            torch.stack(y_train)
        )
        test_data = TensorDataset(
            torch.from_numpy(X_test),
            torch.from_numpy(mask_test),
            torch.stack(y_test)
        )
        valid_data = TensorDataset(
            torch.from_numpy(X_valid),
            torch.from_numpy(mask_valid),
            torch.stack(y_valid)
        )

        batch_size = args.batch_size

        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(
            test_data, shuffle=False, batch_size=batch_size)
        valid_loader = DataLoader(
            valid_data, shuffle=False, batch_size=batch_size)

        return train_loader, valid_loader, test_loader, vocab


class SST2(object):

    def __init__(self) -> None:
        super(SST2, self).__init__()

    def load(self, args):

        train_df = pd.read_csv('data/sst/train.tsv', sep='\t')
        valid_df = pd.read_csv('data/sst/dev.tsv', sep='\t')
        test_df = pd.read_csv('data/sst/test.tsv', sep='\t')

        # TODO: columns
        train_text = train_df['sentence'].tolist()
        train_label = multi_class_process(train_df['label'].tolist(), 2)

        valid_text = valid_df['sentence'].tolist()
        valid_label = multi_class_process(valid_df['label'].tolist(), 2)

        test_text = test_df['sentence'].tolist()

        clean_train = [data_preprocessing(t) for t in train_text]
        clean_valid = [data_preprocessing(t) for t in valid_text]
        clean_test = [data_preprocessing(t) for t in test_text]

        vocab = create_vocab(clean_train)

        clean_train_id = convert2id(clean_train, vocab)
        clean_valid_id = convert2id(clean_valid, vocab)
        clean_test_id = convert2id(clean_test, vocab)

        cti = []
        tl = []
        for i in range(len(clean_train_id)):
            if len(clean_train_id[i]) >= 1:
                cti.append(clean_train_id[i])
                tl.append(train_label[i])
        clean_train_id = cti
        train_label = tl

        max_len = max([len(s) for s in clean_train_id])
        print('max seq length', max_len)

        train_features = Padding(clean_train_id, max_len)
        test_features = Padding(clean_test_id, max_len)
        valid_features = Padding(clean_valid_id, max_len)

        X_train, X_valid, y_train, y_valid = train_features, valid_features, train_label, valid_label
        X_test = test_features

        print('dataset information:')
        print('=====================')
        print('train size', len(X_train))
        print('valid size', len(X_valid))
        print('test size', len(test_features))
        print('=====================')

        train_data = TensorDataset(
            torch.from_numpy(X_train), torch.stack(y_train))
        test_data = TensorDataset(torch.from_numpy(X_test))
        valid_data = TensorDataset(
            torch.from_numpy(X_valid), torch.stack(y_valid))

        batch_size = args.batch_size

        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(
            test_data, shuffle=False, batch_size=batch_size)
        valid_loader = DataLoader(
            valid_data, shuffle=False, batch_size=batch_size)

        return train_loader, valid_loader, test_loader, vocab

    def load_with_masks(self, args):

        train_df = pd.read_csv('data/sst/train.tsv', sep='\t')
        valid_df = pd.read_csv('data/sst/dev.tsv', sep='\t')
        test_df = pd.read_csv('data/sst/test.tsv', sep='\t')

        # TODO: columns
        train_text = train_df['sentence'].tolist()
        train_label = multi_class_process(train_df['label'].tolist(), 2)

        valid_text = valid_df['sentence'].tolist()
        valid_label = multi_class_process(valid_df['label'].tolist(), 2)

        test_text = test_df['sentence'].tolist()

        clean_train = [data_preprocessing(t) for t in train_text]
        clean_valid = [data_preprocessing(t) for t in valid_text]
        clean_test = [data_preprocessing(t) for t in test_text]

        vocab = create_vocab(clean_train)

        clean_train_id = convert2id(clean_train, vocab)
        clean_valid_id = convert2id(clean_valid, vocab)
        clean_test_id = convert2id(clean_test, vocab)

        cti = []
        tl = []
        for i in range(len(clean_train_id)):
            if len(clean_train_id[i]) >= 1:
                cti.append(clean_train_id[i])
                tl.append(train_label[i])
        clean_train_id = cti
        train_label = tl

        max_len = max([len(s) for s in clean_train_id])
        print('max seq length', max_len)

        train_features, train_mask = PadTransformer(clean_train_id, max_len)
        valid_features, valid_mask = PadTransformer(clean_valid_id, max_len)
        test_features, test_mask = PadTransformer(clean_test_id, max_len)

        X_train, mask_train, y_train = train_features, train_mask, train_label
        X_valid, mask_valid, y_valid = valid_features, valid_mask, valid_label
        X_test, mask_test = test_features, test_mask

        print('dataset information:')
        print('=====================')
        print('train size', len(X_train))
        print('valid size', len(X_valid))
        print('test size', len(test_features))
        print('=====================')

        train_data = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(mask_train),
            torch.stack(y_train)
        )
        test_data = TensorDataset(
            torch.from_numpy(X_test),
            torch.from_numpy(mask_test),
        )
        valid_data = TensorDataset(
            torch.from_numpy(X_valid),
            torch.from_numpy(mask_valid),
            torch.stack(y_valid)
        )

        batch_size = args.batch_size

        train_loader = DataLoader(
            train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=1)
        valid_loader = DataLoader(
            valid_data, shuffle=False, batch_size=batch_size)

        return train_loader, valid_loader, test_loader, vocab


class WikiFR(object):

    def __init__(self) -> None:
        super(WikiFR, self).__init__()

    def load(self) -> Dataset:
        return load_dataset('wikipedia', '20200501.fr')


class CoolDataset(Dataset):

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample
