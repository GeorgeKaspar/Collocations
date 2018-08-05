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


def generate_ngram(path, window_size):
    uni = Counter()
    bi = Counter()
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in tqdm(reader):
            text = row[1].decode('utf-8')
            words = map(lambda x: tagger.normal_form(x), re.findall(r'[\w]+', text, re.U))
            uni.update(words)
            for i in range(1, window_size):
                bi.update(zip(words[:-i], words[i:]))
    return uni, bi

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    uni, bi = None, None
    if os.path.exists('uni.pkl'):
        with open('uni.pkl', 'rb') as f, open('bi.pkl', 'rb') as g:
            uni = pickle.load(f)
            bi = pickle.load(g)
    else:
        uni, bi = generate_ngram(args.path, 3)
        with open('uni.pkl', 'wb') as f, open('bi.pkl', 'wb') as g:
            pickle.dump(uni, f)
            pickle.dump(bi, g)

    N = float(sum(uni.values()))
    M = float(sum(bi.values()))
    for pair, value in tqdm(bi.items()):
        if not (tagger.is_part_of_name(pair[0])[1] and tagger.is_part_of_name(pair[1])[1]):
            continue
        mu = (uni[pair[0]] / N) * (uni[pair[1]] / N)
        x = value / M
        sigma = math.sqrt(mu * (1 - mu) / M)
        t = (x - mu) / sigma
        if t > 2.576:
            print(pair[0] + '\t' + pair[1])
