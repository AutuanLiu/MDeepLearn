#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
   File Name：text
   Description :  文本分类问题 SVD and NMF
   都是可以用来降维或者特征提取的
   Email : autuanliu@163.com
   Date：18-2-3
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, randomized_svd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 获取数据: 4 个类别, 2034 个样本
cate = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
remove = ('headers', 'footers', 'quotes')
news_train = fetch_20newsgroups(
    data_home='../../datasets/news/',
    subset='train',
    categories=cate,
    remove=remove)
news_test = fetch_20newsgroups(
    data_home='../datasets/news/',
    subset='test',
    categories=cate,
    remove=remove)

# 获取每个样本中的词频
vectorizer = CountVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(news_train.data).todense()
vocab = np.array(vectorizer.get_feature_names())

# 奇异值分解 SVD
U, s, Vh = np.linalg.svd(vectors, full_matrices=False)
print(U, s, Vh)

# 判断是否相等
print(np.allclose(U @ np.diag(s) @ Vh, vectors))

# 判断是否正交
print(np.allclose(U.T @ U, np.eye(*U.shape)))

# 奇异值
plt.figure(1)
plt.plot(s)

#  NMF
# V = W @ H
clf = NMF(n_components=5, random_state=0)
W = clf.fit_transform(vectors)
H = clf.components_

print(W, H)
plt.figure(2)
plt.plot(H[0])

# 判断是否相等
# NMF 的分解是不准确的
print(np.allclose(W @ H, vectors))

# TF-IDF
vectorizer_tfidf = TfidfVectorizer(stop_words='english')
vectors_tfidf = vectorizer_tfidf.fit_transform(news_train.data)  # (documents, vocab)
W1 = clf.fit_transform(vectors_tfidf)
H1 = clf.components_
# 误差
# Frobenius norm of V−WH
print(clf.reconstruction_err_)

# 第一个组分的重要性
plt.figure(3)
plt.plot(H1[0])

# 随机 SVD
u, s, v = randomized_svd(vectors, 5)
print(u, s, v)

plt.show()
