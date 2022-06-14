
# Usage instructions

* `Dataset`: (str) `"AGNews" || "DBpedia" || "IMDb" || "SST"`
* `Mode`: (str) `"LSTM" || "LSTMAL" || "Transformer" || "TransformerAL"`
* `Parameters`:

        "activation": "tanh" || "sigmoid" (more option in utils.py--get_act()),
        "batch_size": 256,
        "class_num": 4 (agnews) || 14 (dbpedia) || 2 (sst) || 2 (IMDB),
        "embedding_dim": 300,
        "epochs": 50,
        "hidden_dim": 512,
        "label_dim": 128,
        "lr": 0.0001,
        "max_len": 256,
        "nhead": 6,
        "nlayers": 2,
        "one_hot_label": true,
        "pretrained": "glove" || "fasttext",
        "ramdom_label": false (only used in generalization test),
        "vocab_size": 30000
