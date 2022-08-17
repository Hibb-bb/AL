import argparse
from datasets import load_dataset
from nltk.corpus import stopwords
from torch.utils.data import DataLoader
from utils import *
from model import Model
from tqdm import tqdm

stop_words = set(stopwords.words('english'))


def get_args():

    parser = argparse.ArgumentParser('AL training')

    # model param
    parser.add_argument('--emb-dim', type=int,
                        help='word embedding dimension', default=300)
    parser.add_argument('--label-emb', type=int,
                        help='label embedding dimension', default=128)
    parser.add_argument('--l1-dim', type=int,
                        help='lstm1 hidden dimension', default=128)

    parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)
    parser.add_argument('--max-len', type=int, help='max input length', default=200)
    parser.add_argument('--dataset', type=str, default='ag_news', choices=['ag_news', 'dbpedia_14', 'banking77', 'emotion', 'rotten_tomatoes','imdb', 'clinc_oos', 'yelp_review_full'])
    parser.add_argument('--word-vec', type=str, default='glove')

    # training param
    parser.add_argument('--lr', type=float, help='lr', default=0.001)
    parser.add_argument('--batch-size', type=int, help='batch-size', default=16)
    parser.add_argument('--one-hot-label', type=bool,
                        help='if true then use one-hot vector as label input, else integer', default=True)
    parser.add_argument('--epoch', type=int, default=20)

    # dir param
    parser.add_argument('--save-dir', type=str, default='ckpt/')

    args = parser.parse_args()

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

def train(model, ld, epoch):

    model.train()
    cor, num, tot_loss = 0, 0, 0.0
    b = tqdm(ld)
    for x, y in b:
        x, y = x.cuda(), y.cuda()
        loss = model(x, y)
        tot_loss += loss.item()
        pred = model.inference(x)
        cor += (pred.argmax(-1) == y).sum().item()
        num += x.size(0)
        
        b.set_description(f'Acc {100*cor/num} ({cor}/{num})')
    print('Train Epoch', epoch, 'Acc', 100*cor/num, 'Loss', tot_loss/len(ld))

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
    model = Model(args.emb_dim, args.l1_dim, args.l1_dim, class_num, len(vocab), word_vec)
    model = model.cuda()

    print('Start Training')

    for i in range(20):
        train(model, train_loader, i)
        best_acc = test(model, test_loader, i, best_acc, args.save_dir+f'/{args.dataset}.pt')

    print('Best acc', best_acc)

main()
