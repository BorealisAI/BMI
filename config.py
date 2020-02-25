# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# ------------------------- PATH ---------------------------
ROOT_DIR = "."
DATA_DIR = "%s/data" % ROOT_DIR
PENN_DATA_DIR = "%s/penn" % DATA_DIR
WIKI_DATA_DIR = "%s/wikitext-2" % DATA_DIR

CHECKPOINT_DIR = "%s/checkpoint" % ROOT_DIR

# ------------------------- DATA ---------------------------

DATASET = {
    "penn": PENN_DATA_DIR,
    "wiki": WIKI_DATA_DIR,
}

HPARAMS = {
    "penn": {
        "embed_size": 280,
        "hidden_size": 960,
        "last_hidden_size": 620,
        "num_layers": 3,
        "dropout": 0.4,
        "dropouth": 0.225,
        "dropouti": 0.4,
        "dropoute": 0.1,
        "dropoutl": 0.29,
        "wdrop": 0.5,
        "n_experts": 15,
        "tie_weights": True,
        "num_epochs": 1000,
        "bsz1": 12,
        "small_bsz1": 12,
        "bsz2": 12,
        "log_interval": 200,
        "lr": 20.0,
        "wdecay": 1.2e-6,
        "single_gpu": True,
        "rml": False,
        "contrastive1": False,
        "contrastive2": False,
        "contrastive2_rl": False,
        "c_lr": 2e-4,
        "c_wdecay": 1e-6,
        "weight1": 0.1,
        "weight2": 1.0,
        "tau": 1.0,
        "geo_s_p": 0.3,
        "mlp_params": {
            "hidden_state": 300,
            "hidden_layers": 1,
            "hidden_dropout": 0.3,
            "input_dropout": 0.1,
            "bidirectional": True,
        },
    },
    "wiki": {
        "embed_size": 300,
        "hidden_size": 1150,
        "last_hidden_size": 650,
        "num_layers": 3,
        "dropout": 0.4,
        "dropouth": 0.2,
        "dropouti": 0.55,
        "dropoute": 0.1,
        "dropoutl": 0.29,
        "wdrop": 0.5,
        "n_experts": 15,
        "tie_weights": True,
        "num_epochs": 1000,
        "bsz1": 15,
        "small_bsz1": 5,
        "bsz2": 10,
        "log_interval": 200,
        "lr": 15.0,
        "wdecay": 1.2e-6,
        "single_gpu": True,
        "rml": False,
        "contrastive1": False,
        "contrastive2": False,
        "contrastive2_rl": False,
        "c_lr": 2e-4,
        "c_wdecay": 1e-6,
        "weight1": 0.02,
        "weight2": 0.05,
        "tau": 1.0,
        "geo_s_p": 0.3,
        "mlp_params": {
            "hidden_state": 100,
            "hidden_layers": 1,
            "hidden_dropout": 0.1,
            "input_dropout": 0.1,
            "bidirectional": True,
        },
    },
}


# ------------------------- PARAM --------------------------

RANDOM_SEED = 8888
