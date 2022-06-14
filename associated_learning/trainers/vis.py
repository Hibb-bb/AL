from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import  pairwise 


def distance(x, y, class_num):
    x = np.stack(x, axis=0)
    inter_dis=[]
    outer_dis=[]
    print(class_num)
    print(x.shape, y.shape)
    for i in range(class_num):
        x_c = x[np.where(y==i)]
        x_o = x[np.where(y!=i)]
        inter = pairwise.pairwise_distances(x_c, metric='euclidean')
        outer = pairwise.pairwise_distances(x_c, x_o, metric='euclidean')
        inter = np.mean(np.mean(inter, axis=0), axis=0)
        outer = np.mean(np.mena(outer, axis=0), axis=0)
        inter_dis.append(inter)
        outer_dis.append(outer)
    print('inter class distance', inter_dis)
    print('outer class distance', outer_dis)
    return inter_dis, outer_dis


def tsne(x,y, class_num, save_dir):
    inter, outer = distance(x,y,class_num)
    return 0
    '''
    x: (n_sample, n_feature)
    y: (n_sample, )
    class_num: int
    '''
    x_emb = TSNE(n_components=2, random_state=0, perplexity=100).fit_transform(x)
    x_emb_x = x_emb[:,0]
    x_emb_y = x_emb[:,1]
    # cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3','tab10', 'tab20', 'tab20b', 'tab20c']
    colors = [plt.cm.tab20(i) for i in range(len(plt.cm.tab20))] + [plt.cm.tab20b(i) for i in range(len(plt.cm.tab20b))]+ [plt.cm.tab20c(i) for i in range(len(plt.cm.tab20c))]+ [plt.cm.tab10(i) for i in range(len(plt.cm.tab10))]+ [plt.cm.paired(i) for i in range(len(plt.cm.paired))]
    plt.scatter(x_emb_x, x_emb_y, c=colors[:class_num])
    plt.savefig(save_dir)
