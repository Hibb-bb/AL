
# Associated_Learning

# AL

AL (Associated Learning, [ICLR 2022](https://in.ncu.edu.tw/~hhchen/academic_works/wu22-associated.pdf) and [NECO 2021](https://github.com/SamYWK/Associated_Learning)) decomposes a neural network's layers into small components such that **each component has a local objective function**.  As a result, **each layer can be trained independently and simultaneously** (in a pipeline fashion).  AL's predictive power is comparable to (and frequently better than) end-to-end backpropagation.

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets

* For AGNews and DBpedia, dataset will be automatically downloaded during the training.
* For SST-2, please download the dataset from [GLUE Benchmark](https://gluebenchmark.com/tasks) and put the files into `./data/sst2/`.

## Execution

We use json file for the configuration. Before running the code, please check [`hyperparameters.json`](configs/) and select proper parameters.

Options for hyperparameters:

   * Datasets(training dataset): AGNews, DBpedia, IMDB, SST.
   * Mode(model structure and propagation method): LSTMAL, LSTM, Transformer, TransformerAL
   * activation(activation used in AL bridge function): please see utils.py - get_act()

Then just simply run:

```bash
python -m associated_learning.main
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
