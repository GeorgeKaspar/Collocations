import numpy as np
import csv
import re
import sys
from nltk import ngrams
from collections import defaultdict, Counter
from tqdm import tqdm
import sys
import math
import pickle
import os
import pymorphy2
from tagger import tagger
csv.field_size_limit(sys.maxsize)


def generate_ngram(path):
    uni = Counter()
    bi = Counter()
    tri = Counter()
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in tqdm(reader):
            text = row[1].decode('utf-8')
            words = map(lambda x: tagger.normal_form(x), re.findall(r'[\w]+', text, re.U))
            uni.update(words)
            bi.update(zip(words[:-1], words[1:]))
            tri.update(zip(words[:-2], words[1:-1], words[2:]))
    return uni, bi, tri


class TrigramLanguageModel(object):
    def __init__(self, uni, bi, tri, laplace=True):
        self.uni = uni
        self.bi = bi
        self.tri = tri
        self.N = sum(self.uni.values())
        self.M = sum(self.bi.values())
        self.K = sum(self.tri.values())

        self.uni_num = len(self.uni)
        self.bi_num = len(self.bi)
        self.tri_num = len(self.tri)
        self.laplace = laplace

    def predict(self, phrase):
        predict_single = (lambda x: self.predict_single_laplace(x)) if self.laplace else: (lambda x: self.predict_single_nonlaplace(x))
        words = map(lambda x: tagger.normal_form(x), re.findall(r'[\w]+', text, re.U))
        return map(predict_single, zip(words[:-2], words[1:-1], words[2:]))

    def predict_single_nonlaplace(self, trigram):
        return (self.tri.get((trigram[0], trigram[1], trigram[2]), 0) + self.tri.get((trigram[1], trigram[0], trigram[2]), 0)) / (self.bi.get((trigram[0], trigram[1]), 0) + self.bi.get((trigram[1], trigram[0]), 0))

    def predict_single_laplace(self, trigram):
        return (self.tri.get((trigram[0], trigram[1], trigram[2]), 0) + self.tri.get((trigram[1], trigram[0], trigram[2]), 0) + 1) / (self.bi.get((trigram[0], trigram[1]), 0) + self.bi.get((trigram[1], trigram[0]), 0) + self.bi_num)
