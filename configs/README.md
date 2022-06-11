
# Usage instructions

* `Dataset`: (str) `"AGNews" || "DBpedia" || "IMDb" || "SST"`
* `Mode`: (str) `"LSTM" || "LSTMAL" || "Transformer" || "TransformerAL"`
* `Parameters`:

        "activation": "tanh" || "sigmoid",
        "batch_size": 256,
        "class_num": 4 (agnews) || 14 (dbpedia) || 2 (sst),
        "embedding_dim": 300,
        "epochs": 50,
        "hidden_dim": 512,
        "label_dim": 128,
        "lr": 0.001,
        "max_len": 256,
        "nhead": 6,
        "nlayers": 2,
        "one_hot_label": true,
        "pretrained": "glove" || "fasttext",
        "ramdom_label": false,
        "vocab_size": 30000
