#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：Decorator
   Description :  装饰器学习
   1. 本质上，decorator 就是一个返回函数的高阶函数
   2. 如果 decorator 本身需要传入参数，那就需要编写一个返回 decorator 的高阶函数
   3. Python 的 decorator 可以用函数实现，也可以用类实现
   4. 装饰器不仅可以是函数，还可以是类，相比函数装饰器，类装饰器具有灵活度大、高内聚、封装性等优点。使用类装饰器主要依靠类的__call__ 方法，当使用 @ 形式将装饰器附加到函数上时，就会调用此方法。
   Email : autuanliu@163.com
   Date：2018/3/19
"""

import time
from functools import wraps

import numpy as np


# 不带参数的装饰器
def exe_add(func):
    @wraps(func)
    def now(*args, **kwargs):
        localtime = time.asctime(time.localtime(time.time()))
        print(f'The execution time is: {localtime}')
        return func(*args, **kwargs)

    return now


# 带有参数的装饰器
def exe(author='autuanliu'):
    assert type(author) is str, 'The args shoud be str!'

    def decorator(func):
        @wraps(func)
        def now(*args, **kwargs):
            localtime = time.asctime(time.localtime(time.time()))
            print(f'The execution time is: {localtime}')
            print(f'The author is: {author}')
            return func(*args, **kwargs)

        return now

    return decorator


# 不带参数类装饰器
class Foo():
    def __init__(self, func):
        self._func = func

    def __call__(self, *args, **kwargs):
        print('class decorator without args')
        self._func(*args, **kwargs)


@exe_add
def addxy(*args, **kwargs):
    res = np.sum(*args, **kwargs)
    print(f'The result is: {res}')


@exe(author='liu')
def max_xy(*args, **kwargs):
    res = np.max(*args, **kwargs)
    print(f'The result is: {res}')


@exe()
def std_xy(*args, **kwargs):
    res = np.std(*args, **kwargs)
    print(f'The result is: {res}')


@Foo
def std_xy1(*args, **kwargs):
    res = np.std(*args, **kwargs)
    print(f'The result is: {res}')


addxy([1, 2, 3])
addxy(np.arange(9).reshape(-1, 3), axis=1, dtype=np.int32)

max_xy([1, 2, 3])
max_xy(np.arange(9).reshape(-1, 3), axis=1)

std_xy(np.arange(9).reshape(-1, 3), axis=1)
std_xy1(np.arange(9).reshape(-1, 3), axis=1)
