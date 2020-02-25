# Language Modelling with BMI Regularizer

Better Long-Range Dependency By Bootstrapping A Mutual Information Regularizer

Paper published in AISTATS 2020: [arxiv.org/abs/1905.11978](https://arxiv.org/abs/1905.11978)

This implementation is based on PyTorch 0.4.1.

### Data

```
./get_data.sh
```

### Training and Evaluation (Single GPU)

Each run will create a folder in `.checkpoint`, which can be re-used later.

#### Penn Treebank

BMI-base (i.e. next sentence prediction)

```
python train_lm.py --contrastive1 --save NSP
```

BMI-full (i.e. RAML), after the completion of the training of BMI-base

```
python train_lm.py --contrastive1 --rml --save RAML --checkpoint PATH_TO_NSP
```

Finetune BMI-base
```
python finetune.py --contrastive1 --lr 25.0 --save PATH_TO_NSP
```

Finetune BMI-full
```
python finetune.py --contrastive1 --rml --lr 25.0 --save PATH_TO_RAML
```

#### WikiText-2

BMI-base (i.e. next sentence prediction)

```
python train_lm.py --data_name wiki --bsz1 15 --small_bsz1 5 --bsz2 10 --contrastive1 --save NSP
```

BMI-full (i.e. RAML), after the completion of the training of BMI-base

```
python train_lm.py --data_name wiki --bsz1 15 --small_bsz1 5 --bsz2 10 --contrastive1 --rml --save RAML --load PATH_TO_NSP
```

Finetune BMI-base
```
python finetune.py --data_name wiki --bsz1 15 --small_bsz1 5 --bsz2 10 --contrastive1 --lr 20.0 --save PATH_TO_NSP
```

Finetune BMI-full
```
python finetune.py --data_name wiki --bsz1 15 --small_bsz1 5 --bsz2 10 --contrastive1 --rml --lr 20.0 --save PATH_TO_RAML
```

### Acknowledgements:

A large portion of this repo is borrowed from https://github.com/zihangdai/mos

### Cite

If you found this codebase or our work useful, please cite:

```
@InProceddings{cao2020better,
    author = {Yanshuai, Cao and Xu, Peng},
    title = {Better Long-Range Dependency By Bootstrapping A Mutual Information Regularizer}
    booktitle = {The 23rd International Conference on Artificial Intelligence and Statistics (AISTATS 2020)},
    month = {June},
    year = {2020},
    publisher = {PMLR}
}
```

### License

Copyright (c) 2018-present, Royal Bank of Canada. All rights reserved.
This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
