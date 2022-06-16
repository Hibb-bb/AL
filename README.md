
# Associated Learning

This repo contains the Pytorch code used to train the models used in the paper:<br>
**Associated Learning: a Methodology to Decompose End-to-End Backpropagation on CNN, RNN, and Transformer**, presented at ICLR 2022.

AL (Associated Learning, [ICLR 2022](https://in.ncu.edu.tw/~hhchen/academic_works/wu22-associated.pdf) and [NECO 2021](https://github.com/SamYWK/Associated_Learning)) decomposes a neural network's layers into small components such that **each component has a local objective function**.  As a result, **each layer can be trained independently and simultaneously** (in a pipeline fashion).  AL's predictive power is comparable to (and frequently better than) end-to-end backpropagation.

## Requirements

*Using virtual environment is recommended.* <br>
*During the experiment, we mostly ran our code on RTX3090, but 1070 should be enough.*

```bash
mkdir data
mkdir data/ckpt
pip install -r requirements.txt
```

## Datasets

* For AGNews and DBpedia, dataset will be automatically downloaded during the training.
* For SST-2, please download the dataset from [GLUE Benchmark](https://gluebenchmark.com/tasks) and put the files into `./data/sst2/`.
* For IMDB, please download the dataset from [Here](https://drive.google.com/file/d/15bZOVHo6KESFl-WKnElEjcM_qZkjip21/view?usp=sharing), and unzip the file into `data/`(should be like `data/imdb/`), we use the standard train-test split of IMDB dataset.
* To evaluate the performance of SST-2, a prediction file [`data/sst2/SST-2.tsv`](data/sst2/) will be generated, please submit it along with the GLUE submission format.

## Word Embeddings

* We uses pretrained embeddings in our experiments, please download GloVe, Fasttext with the following commands. 

```bash
wget https://nlp.stanford.edu/data/glove.6B.zip
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip glove.6B.zip
unzip wiki-news-300d-1M.vec.zip
```

## Execution

We use json file for the configuration. Before running the code, please check [`hyperparameters.json`](configs/) and select proper parameters.

Options for hyperparameters:

   * Datasets(training dataset): AGNews, DBpedia, IMDB, SST.
   * Mode(model structure and propagation method): LSTMAL, LSTM, Transformer, TransformerAL
   * activation(activation used in AL bridge function): please see utils.py - get_act()

Then just simply run:

```bash
python3 -m associated_learning.main --config-file configs/hyperparameters.json
```

## Citation

    @inproceedings{
      wu2022associated,
      title={Associated Learning: an Alternative to End-to-End Backpropagation that Works on {CNN}, {RNN}, and Transformer},
      author={Dennis Y.H. Wu and Dinan Lin and Vincent Chen and Hung-Hsuan Chen},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=4N-17dske79}
    }
