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


class PersonTagger:
    stemmer = pymorphy2.MorphAnalyzer()
    proba_th = 0.4
    d = {}

    def is_part_of_name(self, word):
        ans = self.d.get(word, None)
        if ans is not None:
            return ans
        res = self.stemmer.parse(word)[0]
        if res.normal_form is None:
            res.normal_form = word
        if ('Name' in res.tag or 'Surn' in res.tag or 'Patr' in res.tag) and res.score >= self.proba_th:
            ans = (res.normal_form, True)
        else:
            ans = (res.normal_form, False)
        self.d[word] = ans
        self.d[ans[0]] = ans
        return ans

    def normal_form(self, word):
        return self.is_part_of_name(word)[0]

tagger = PersonTagger()
