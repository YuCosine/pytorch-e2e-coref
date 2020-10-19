from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import codecs
import collections
import datetime
import json
import math
import shutil
import sys
import subprocess

import numpy as np
import pyhocon


def TensorboardWriter(save_path):
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(save_path, comment="Unmt")

def initialize_from_env(eval_test=False):
    if "GPU" in os.environ:
        set_gpus(int(os.environ["GPU"]))

    if len(sys.argv) >= 2:
        name = sys.argv[1]
    else:
        name = 'debug'
    if len(sys.argv) >= 3:
        mode = sys.argv[2]
    else:
        mode = 'train'

    print("Running experiment: {}".format(name))

    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[name]

    if mode == 'debug':
        name += '_debug'

    config["log_dir"] = os.path.join(config["log_dir"], name)
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
    print(f"Results saved to {config['log_dir']}")

    config['timestamp'] = datetime.datetime.now().strftime('%m%d-%H%M%S')

    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def set_gpus(*gpus):
    # pass
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))  



def set_log_file(fname):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything

    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def copy_checkpoint(source, target):
    for ext in (".index", ".data-00000-of-00001"):
        shutil.copyfile(source + ext, target + ext)


def flatten(l):
  return [item for sublist in l for item in sublist]

def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def load_char_dict(char_vocab_path):
    vocab = [u"<unk>"]
    with codecs.open(char_vocab_path, encoding="utf-8") as f:
        vocab.extend(l.strip() for l in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(vocab)})
    return char_dict


def maybe_divide(x, y):
    return 0 if y == 0 else x / float(y)


class RetrievalEvaluator(object):
    def __init__(self):
        self._num_correct = 0
        self._num_gold = 0
        self._num_predicted = 0

    def update(self, gold_set, predicted_set):
        self._num_correct += len(gold_set & predicted_set)
        self._num_gold += len(gold_set)
        self._num_predicted += len(predicted_set)

    def recall(self):
        return maybe_divide(self._num_correct, self._num_gold)

    def precision(self):
        return maybe_divide(self._num_correct, self._num_predicted)

    def metrics(self):
        recall = self.recall()
        precision = self.precision()
        f1 = maybe_divide(2 * recall * precision, precision + recall)
        return recall, precision, f1


class EmbeddingDictionary(object):
    def __init__(self, info, normalize=True, maybe_cache=None):
        self._size = info["size"]
        self._normalize = normalize
        self._path = info["path"]
        if maybe_cache is not None and maybe_cache._path == self._path:
            assert self._size == maybe_cache._size
            self._embeddings = maybe_cache._embeddings
        else:
            self._embeddings = self.load_embedding_dict(self._path)

    @property
    def size(self):
        return self._size

    def load_embedding_dict(self, path):
        print("Loading word embeddings from {}...".format(path))
        default_embedding = np.zeros(self.size)
        embedding_dict = collections.defaultdict(lambda: default_embedding)
        if len(path) > 0:
            vocab_size = None
            with open(path) as f:
                for i, line in enumerate(f.readlines()):
                    word_end = line.find(" ")
                    word = line[:word_end]
                    embedding = np.fromstring(line[word_end + 1:], np.float32, sep=" ")
                    assert len(embedding) == self.size
                    embedding_dict[word] = embedding
            if vocab_size is not None:
                assert vocab_size == len(embedding_dict)
            print("Done loading word embeddings.")
        return embedding_dict

    def __getitem__(self, key):
        embedding = self._embeddings[key]
        if self._normalize:
            embedding = self.normalize(embedding)
        return embedding

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        else:
            return v

