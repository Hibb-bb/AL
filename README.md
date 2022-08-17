
# Associated Learning
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <br>
This repo contains the Pytorch code used to train the models used in the paper:<br>
**Associated Learning: a Methodology to Decompose End-to-End Backpropagation on CNN, RNN, and Transformer**, presented at ICLR 2022.

AL (Associated Learning, [ICLR 2022](https://in.ncu.edu.tw/~hhchen/academic_works/wu22-associated.pdf) and [NECO 2021](https://github.com/SamYWK/Associated_Learning)) decomposes a neural network's layers into small components such that **each component has a local objective function**.  As a result, **each layer can be trained independently and simultaneously** (in a pipeline fashion).  AL's predictive power is comparable to (and frequently better than) end-to-end backpropagation.

## Datasets

* For AGNews and DBpedia, dataset will be automatically downloaded during the training.
* For SST-2, please download the dataset from [GLUE Benchmark](https://gluebenchmark.com/tasks) and put the files into `./data/sst2/`.
* For IMDB, please download the dataset from [Here](https://drive.google.com/file/d/1GRyOQs6TT0IXKDyha6zNjinmvREKyeuV/view?usp=sharing)
* To evaluate the performance of SST-2, a prediction file [`data/sst2/SST-2.tsv`](data/sst2/) will be generated, please submit it along with the GLUE submission format.
* Actually, we includes more datasets in this repo, including: `ag_news, dbpedia_14, banking77, emotion, rotten_tomatoes, imdb, clinc_oos, yelp_review_full, sst2`
* For `banking77`, please indicate a longer training epoch e.g. 300.
 
## Word Embeddings

* We uses pretrained embeddings in our experiments, please download GloVe, Fasttext with the following commands. 

## Requirements

*During the experiment, we mostly ran our code on RTX3090, but 1070 should be enough.*<br>
*Tested under Ubuntu with Python 3.8.*

For this repo, I recommend using virtualenv
Please run <br> 
```bash
mkdir ckpt
virtualenv ./env
source ./env/bin/activate
pip3 install -r requirements.txt
wget https://nlp.stanford.edu/data/glove.6B.zip
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip glove.6B.zip
unzip wiki-news-300d-1M.vec.zip
```

## Execution for Distributed training setup
 I think that pytorch has some bug when using a single optimizer for two seperated gradient path (the computation result can be incorrect), so I suggest running with the following commands <br> In order to ensure a more stable training results, please run <br>
 ```bash
 python3 dis_train.py --dataset <DATASET>
 ``` 
 For Transformer training, please run <br>
 ```bash
python3 dis_train_tr.py --dataset <DATASET>
 ```
  
(IMDB is under construction)
To run IMDB, please run
```bash
python3 imdb_al.py
```

I highly recommend building your code upon `distributed_model.py` for people whom whats to do further studies.

Please remember to change hyperparameter when training on different dataset
If max_len is 1000, it means that we use the max length in the training data.

| Dataset          | l1_dim | max_len |
|------------------|--------|---------|
| ag_news          | 300    | 1000    |
| dbpedia_14       | 300    | 500     |
| banking77        | 256    | 1000    |
| emotion          | 256    | 200     |
| rotten_tomatoes  | 256    | 100     |
| yelp_review_full | 256    | 400     |

## Citation

If you find AL useful in your research, you can cite our work in your research, thank you.

    @inproceedings{
      wu2022associated,
      title={Associated Learning: an Alternative to End-to-End Backpropagation that Works on {CNN}, {RNN}, and Transformer},
      author={Dennis Y.H. Wu and Dinan Lin and Vincent Chen and Hung-Hsuan Chen},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=4N-17dske79}
    }
