#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：normalizedFunc
   Description :  实现几种常用的标准化函数
   Email : autuanliu@163.com
   Date：2017/12/10
"""

import numpy as np


# min-max 标准化
# 按列
def min_max_normalized(data):
    """标准化

    最大最小值标准化

    Parameters
    ----------
    :param data: numpy array
        待标准化的数据

    Returns
    -------
    :return res:
       标准化结果

    Examples
    --------
    >>> X = np.array([-2, -3, 0, -1, 2])
    >>> res = min_max_normalized(X)
    >>> print(res)
    [0.2 0.  0.6 0.4 1. ]
    """
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)
