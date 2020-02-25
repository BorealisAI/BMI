# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from utils.data_utils import DataLoader
from models.language_models import LanguageModel
import config
import os
import argparse

import numpy as np
import time
import shutil
import torch


def main(args):
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True

    data_path = config.DATASET[args.data_name]
    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")
    test_path = os.path.join(data_path, "test.txt")
    vocab_path = os.path.join(data_path, "vocab.pkl")
    max_len_delta = 40 if "penn" in args.data_name else 20
    train = DataLoader(train_path, vocab_path, max_len_delta, args.mode)
    valid = DataLoader(valid_path, vocab_path, max_len_delta, args.mode)
    test = DataLoader(test_path, vocab_path, max_len_delta, args.mode)

    save_path = '{}-{}-{}'.format(args.save, args.data_name, time.strftime("%Y%m%d-%H%M%S"))
    save_path = os.path.join(config.CHECKPOINT_DIR, save_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    script_path = os.path.join(save_path, 'scripts')
    scripts_to_save = [
        'train_lm.py', 'models/language_models.py', 'models/base_network.py',
        'utils/data_utils.py', 'config.py']

    if not os.path.exists(script_path):
        os.mkdir(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(script_path, os.path.basename(script))
            shutil.copyfile(script, dst_file)

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(save_path, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    if "penn" in args.data_name:
        hparams = config.HPARAMS["penn"]
    else:
        hparams = config.HPARAMS[args.data_name]

    hparams["bsz1"] = args.bsz1
    hparams["small_bsz1"] = args.small_bsz1
    hparams["bsz2"] = args.bsz2
    hparams["contrastive1"] = args.contrastive1
    hparams["contrastive2"] = args.contrastive2
    hparams["contrastive2_rl"] = args.contrastive2_rl
    hparams["rml"] = args.rml

    kwargs = {
        "train": train,
        "valid": valid,
        "test": test,
        "save_path": save_path,
        "data": None,
        "hparams": hparams
    }

    logging(str(kwargs))

    lm = LanguageModel(**kwargs)
    lm.init()
    if args.checkpoint != 'NA':
        lm.load(args.checkpoint)
    try:
        val_epoch, val_loss, val_acc1, val_acc2 = lm.fit()
        logging("val epoch: {}".format(val_epoch))
        logging("val loss : {}".format(val_loss))
        logging("val ppl  : {}".format(np.exp(val_loss)))
        logging("val acc1 : {}".format(val_acc1))
        logging("val acc2 : {}".format(val_acc2))
    except KeyboardInterrupt:
        logging("Exiting from training early")

    lm.load(lm.save_path)
    test_loss, test_acc1, test_acc2 = lm.evaluate(lm.test_dataloader, 1, args.bsz2)

    logging("test loss: {}".format(test_loss))
    logging("test ppl : {}".format(np.exp(test_loss)))
    logging("test acc1: {}".format(test_acc1))
    logging("test acc2: {}".format(test_acc2))


def add_args(parser):
    parser.add_argument('--data_name', type=str, default='penn',
                        help='data name')

    parser.add_argument('--mode', type=str, default='sent',
                        help='sentence or chunk level')

    parser.add_argument('--bsz1', type=int, default=12,
                        help='batch size for lm')

    parser.add_argument('--small_bsz1', type=int, default=12,
                        help='batch size for lm')

    parser.add_argument('--bsz2', type=int, default=12,
                        help='batch size for discriminator')

    parser.add_argument('--contrastive1', default=False, action="store_true",
                        help="enable contrastive 1")

    parser.add_argument('--contrastive2', default=False, action="store_true",
                        help="enable contrastive 2")

    parser.add_argument('--contrastive2_rl', default=False, action="store_true",
                        help="enable contrastive 2 policy gradient term")

    parser.add_argument('--rml', default=False, action="store_true",
                        help="enable rml")

    parser.add_argument('--save', type=str, default='cLM',
                        help='directory name to save')

    parser.add_argument('--checkpoint', type=str, default='NA',
                        help='directory to load checkpoint')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
