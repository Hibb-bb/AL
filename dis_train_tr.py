import argparse
from datasets import load_dataset
from nltk.corpus import stopwords
from torch.utils.data import DataLoader
from utils import *
# from model import Model
from distributed_model import TransModel
from tqdm import tqdm
import os

stop_words = set(stopwords.words('english'))


def get_args():

    parser = argparse.ArgumentParser('AL training')

    # model param
    parser.add_argument('--emb-dim', type=int,
                        help='word embedding dimension', default=300)
    parser.add_argument('--label-emb', type=int,
                        help='label embedding dimension', default=128)
    parser.add_argument('--l1-dim', type=int,
                        help='lstm1 hidden dimension', default=256)

    parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)
    parser.add_argument('--max-len', type=int, help='max input length', default=200)
    parser.add_argument('--dataset', type=str, default='ag_news', choices=['ag_news', 'dbpedia_14', 'banking77', 'emotion', 'rotten_tomatoes','imdb', 'clinc_oos', 'yelp_review_full', 'sst2'])
    parser.add_argument('--word-vec', type=str, default='glove')

    # training param
    parser.add_argument('--lr', type=float, help='lr', default=0.001)
    parser.add_argument('--batch-size', type=int, help='batch-size', default=64)
    parser.add_argument('--one-hot-label', type=bool,
                        help='if true then use one-hot vector as label input, else integer', default=True)
    parser.add_argument('--epoch', type=int, default=20)

    # dir param
    parser.add_argument('--save-dir', type=str, default='./ckpt-transformer/')

    args = parser.parse_args()

    try:
        os.mkdir(args.save_dir)
    except:
        pass 

    return args

def get_data(args):

    if args.dataset != 'imdb':

        train_data = load_dataset(args.dataset, split='train')
        test_data = load_dataset(args.dataset, split='test')

        if args.dataset == 'dbpedia_14':
            tf = 'content'
            class_num = 14
        elif args.dataset == 'ag_news':
            tf = 'text'
            class_num = 4
        elif args.dataset == 'banking77':
            tf = 'text'
            class_num = 77
        elif args.dataset == 'emotion':
            tf = 'text'
            class_num = 6
        elif args.dataset == 'rotten_tomatoes':
            tf = 'text'
            class_num = 2
        elif args.dataset == 'yelp_review_full':
            tf = 'text'
            class_num = 5
        elif args.dataset == 'sst2':
            tf = 'sentence'
            class_num = 2
            test_data = load_dataset(args.dataset, split='validation')

        train_text = [b[tf] for b in train_data]
        test_text = [b[tf] for b in test_data]
        train_label = [b['label'] for b in train_data]
        test_label = [b['label'] for b in test_data]
        clean_train = [data_preprocessing(t, True) for t in train_text]
        clean_test = [data_preprocessing(t, True) for t in test_text]

        vocab = create_vocab(clean_train)

    else:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        class_num = 2
        df = pd.read_csv('./IMDB_Dataset.csv')
        df['cleaned_reviews'] = df['review'].apply(data_preprocessing, True)
        corpus = [word for text in df['cleaned_reviews'] for word in text.split()]
        text = [t for t in df['cleaned_reviews']]
        label = []
        for t in df['sentiment']:
            if t == 'negative':
                label.append(1)
            else:
                label.append(0)
        vocab = create_vocab(corpus)
        clean_train, clean_test, train_label, test_label = train_test_split(text, label, test_size=0.2)

    trainset = Textset(clean_train, train_label, vocab, args.max_len)
    testset = Textset(clean_test, test_label, vocab, args.max_len)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn = trainset.collate, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, collate_fn = testset.collate)

    return train_loader, test_loader, class_num, vocab


def sci_not(num):
    return format(num,'.1E')

def train(model, ld, epoch):

    model.train()
    cor, num, tot_loss = 0, 0, [0.0]*6
    b = tqdm(ld)
    for step, (x, y) in enumerate(b):
        x, y = x.cuda(), y.cuda()
        losses = model(x, y)
        for _i, l in enumerate(losses):
            tot_loss[_i] += l.item()
        pred = model.inference(x)
        cor += (pred.argmax(-1) == y).sum().item()
        num += x.size(0)
        
        emb_ae, emb_as, l1_ae, l1_as, l2_ae, l2_as = round(tot_loss[0]/(step+1), 4), round(tot_loss[1]/(step+1), 4), round(tot_loss[2]/(step+1), 4), round(tot_loss[3]/(step+1), 4), round(tot_loss[4]/(step+1), 4), round(tot_loss[5]/(step+1), 4),

        b.set_description(f'Train {epoch} | Acc {100*cor/num} ({cor}/{num}) | EMB {emb_ae} + {emb_as} | L1 {l1_ae} + {l1_as} | L2 {l2_ae} + {l2_as}')

def predicting_for_sst(args, model, vocab):

    test_data = load_dataset('sst2', split='test')
    test_text = [b['sentence'] for b in test_data]
    test_label = [b['label'] for b in test_data]
    clean_test = [data_preprocessing(t, True) for t in test_text]
    
    testset = Textset(clean_test, test_label, vocab, args.max_len)
    test_loader = DataLoader(testset, batch_size=1, collate_fn = testset.collate)

    all_pred = []
    all_idx = []
    for i, (x, y) in enumerate(test_loader):
        x = x.cuda()
        pred = model.inference(x).argmax(1).squeeze(0)
        all_pred.append(pred.item())
        all_idx.append(i)
    
    pred_file = {'index':all_idx, 'prediction':all_pred}
    output = pd.DataFrame(pred_file)
    output.to_csv('SST-2.tsv', sep='\t', index=False)

def test(model, ld, epoch, best_acc, ckpt):

    model.eval()
    cor, num = 0, 0
    b = tqdm(ld)
    for x, y in b:
        x, y = x.cuda(), y.cuda()
        pred = model.inference(x)
        cor += (pred.argmax(-1) == y).sum().item()
        num += x.size(0)
        b.set_description(f'Acc {100*cor/num} ({cor}/{num})')

    if 100*cor/num >= best_acc:
        best_acc = 100*cor/num
        torch.save(model.state_dict(), ckpt)

    print('Test Epoch', epoch, 'Acc', 100*cor/num)
    return best_acc

def main():

    best_acc = 0
    args = get_args()
    train_loader, test_loader, class_num, vocab = get_data(args)
    word_vec = get_word_vector(vocab, args.word_vec)
    model = TransModel(vocab_size=len(vocab), emb_dim=args.emb_dim, l1_dim=args.l1_dim, class_num=class_num, word_vec=word_vec, lr=args.lr)
    model = model.cuda()

    print('Start Training')

    for i in range(20):
        train(model, train_loader, i)
        best_acc = test(model, test_loader, i, best_acc, args.save_dir+f'/{args.dataset}.pt')

    print('Best acc', best_acc)

    if args.dataset == 'sst2':
        model.load_state_dict(torch.load(args.save_dir+f'/{args.dataset}.pt'))
        predicting_for_sst(args, model, vocab)

main()
