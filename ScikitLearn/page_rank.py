#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：page_rank
   Description :  page rank 简单实例
   Email : autuanliu@163.com
   Date：2018/3/16
"""

import os
import pickle
from bz2 import BZ2File
from urllib.request import urlopen

import numpy as np
from scipy import sparse
from tqdm import tqdm

# 下载数据
PATH = 'datasets/dbpedia/'
URL_BASE = 'http://downloads.dbpedia.org/3.5.1/en/'
filenames = ["redirects_en.nt.bz2", "page_links_en.nt.bz2"]

for filename in filenames:
    if not os.path.exists(PATH + filename):
        print("Downloading '%s', please wait..." % filename)
        open(PATH + filename, 'wb').write(urlopen(URL_BASE + filename).read())

limit = 119077682    #5000000
redirects_filename = PATH + filenames[0]
page_links_filename = PATH + filenames[1]
DBPEDIA_RESOURCE_PREFIX_LEN = len("http://dbpedia.org/resource/")
SLICE = slice(DBPEDIA_RESOURCE_PREFIX_LEN + 1, -1)


def get_lines(filename):
    return (line.split() for line in BZ2File(filename))


def get_redirect(targ, redirects):
    seen = set()
    while True:
        transitive_targ = targ
        targ = redirects.get(targ)
        if targ is None or targ in seen: break
        seen.add(targ)
    return transitive_targ


def get_redirects(redirects_filename):
    redirects = {}
    lines = get_lines(redirects_filename)
    return {src[SLICE]: get_redirect(targ[SLICE], redirects) for src, _, targ, _ in tqdm(lines, leave=False)}


def add_item(lst, redirects, index_map, item):
    k = item[SLICE]
    lst.append(index_map.setdefault(redirects.get(k, k), len(index_map)))


redirects = get_redirects(redirects_filename)
# Computing the integer index map
index_map = dict()    # links->IDs
lines = get_lines(page_links_filename)
source, destination, data = [], [], []

for l, split in enumerate(lines):
    if l >= limit: 
        break
    add_item(source, redirects, index_map, split[0])
    add_item(destination, redirects, index_map, split[2])
    data.append(1)

n = len(data)

# what type of items are in index_map
print(index_map.popitem())
for page_name, index in index_map.items():
    if index == 9991050:
        print(page_name)

test_inds = [i for i, x in enumerate(source) if x == 9991050]
test_dests = [destination[i] for i in test_inds]

# check which page is the source (has index 9991174)
for page_name, index in index_map.items():
    if index in test_dests:
        print(page_name)

# create a sparse matrix using Scipy's COO format, and that convert it to CSR
X = sparse.coo_matrix((data, (destination,source)), shape=(n,n), dtype=np.float32)
X = X.tocsr()
del(data, destination, source)
names = {i: name for name, i in index_map.items()}

# Save matrix so we don't have to recompute
pickle.dump(X, open(PATH+'X.pkl', 'wb'))
pickle.dump(index_map, open(PATH+'index_map.pkl', 'wb'))
X = pickle.load(open(PATH+'X.pkl', 'rb'))
index_map = pickle.load(open(PATH+'index_map.pkl', 'rb'))
