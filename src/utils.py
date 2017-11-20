"""
Written by Ari Bornstein
"""
import random
import numpy as np

STUDENT = {'name': 'Ari Bornstein',
           'ID': '329710909'}

def read_data(fname):
    data = []
    for line in file(fname):
        label, text = line.strip().lower().split("\t", 1)
        data.append((label, text))
    return data

def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]

TRAIN = [(l, text_to_bigrams(t)) for l, t in read_data(r".\data\train")]
DEV = [(l, text_to_bigrams(t)) for l, t in read_data(r".\data\dev")]
TEST = [(l, text_to_bigrams(t)) for l, t in read_data(r".\data\test")]

UNI_TRAIN = [(l, list(t)) for l, t in read_data("train")]
UNI_DEV = [(l, list(t)) for l, t in read_data("dev")]

from collections import Counter
fc = Counter()
for l, feats in TRAIN:
    fc.update(feats)

uni_fc = Counter()
for l, feats in UNI_TRAIN:
    uni_fc.update(feats)

# 600 most common unigrams in the training set.
uni_vocab = set([x for x, c in uni_fc.most_common(600)])
# 600 most common bigrams in the training set.
vocab = set([x for x, c in fc.most_common(600)])

# label strings to IDs
L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}

# feature strings (bigrams) to IDs
B_F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}

# feature strings (bigrams) to IDs
U_F2I = {f: i for i, f in enumerate(list(sorted(uni_vocab)))}

# IDs to label strings
I2L = {i: l for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}

def get_bigram_data():
    """
    Returns bigram vocab_size, num_langs, train_data, dev_data
    """
    train = [(L2I[item[0]], bigram_to_vec(item[1]))for item in TRAIN]
    dev = [(L2I[item[0]], bigram_to_vec(item[1]))for item in DEV]
    return len(B_F2I), len(L2I), train, dev

def get_unigram_data():
    """
    Returns unigram vocab_size, num_langs, train_data, dev_data
    """
    train = [(L2I[item[0]], unigram_to_vec(item[1]))for item in TRAIN]
    dev = [(L2I[item[0]], unigram_to_vec(item[1]))for item in DEV]
    return len(U_F2I), len(L2I), train, dev

def bigram_to_vec(bigrams):
    """
    Should return a numpy vector of bigrams.
    """
    vocab_count= {v: 0. for v in vocab}
    bigram_count = dict(Counter(bigrams))
    bigram_dict = { k: vocab_count.get(k, 0) + bigram_count.get(k, 0) for k in set(vocab_count)}
    return np.array(bigram_dict.values()) / len(bigrams)

def unigram_to_vec(unigrams):
    """
    Should return a numpy vector of unigrams.
    """
    vocab_count= {v: 0. for v in uni_vocab}
    unigram_count = dict(Counter(unigrams))
    unigram_dict = { k: vocab_count.get(k, 0) + unigram_count.get(k, 0) for k in set(vocab_count)}
    return np.array(unigram_dict.values()) / len(unigrams)

def get_xor_data():
    """
    Get xor data
    """
    xor_in_dim = xor_out_dim = 2
    xor_train_data = [(1, [0, 0]),
                      (0, [0, 1]),
                      (0, [1, 0]),
                      (1, [1, 1])]
    xor_dev_data = ''
    return xor_in_dim, xor_out_dim, xor_train_data, xor_dev_data

# Bigram data for evaluation
TEST_BIGRAMS = [bigram_to_vec(item[1]) for item in TEST]
