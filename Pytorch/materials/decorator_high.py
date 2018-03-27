#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    File Name：decorator_high
    Description :  使用类的高级装饰器
    Email : autuanliu@163.com
    Date：2018/3/23
"""
from abc import abstractmethod, abstractproperty, abstractclassmethod, abstractstaticmethod


# 抽象类定义
class Person():
    def __init__(self, age, gender):
        self.age = age
        self.gender = gender

    @abstractclassmethod
    def run(cls, distance):
        pass

    @abstractstaticmethod
    def do(string):
        pass

    @abstractproperty
    def what(self):
        pass

    @abstractmethod
    def get_name(self):
        pass


class Student(Person):
    def __init__(self, score, *args):
        self.score = score
        super().__init__(*args)

    @classmethod
    def run(cls, distance):
        print(f'Run {distance} m')

    @staticmethod
    def do(string):
        print('Do ', string)

    @property
    def what(self):
        print('Yes')

    def get_age(self):
        return self.age

    @property
    def score(self):
        return self.score

    # @score.setter, 这个需要单层
    def score(self, value):
        assert isinstance(value, float), 'The value should be float!'
        self.score = value

    @property
    def gender(self):
        return self.gender

    # @gender.setter
    def gender(self, string):
        assert isinstance(value, str), 'The value should be string!'
        self.gender = string


# Test example
s1 = Student(95.3, 21, 'm')
s2 = Student(89.3, 18, 'f')

# 属性
s1.what
s2.score
s1.gender

# 普通方法
print('age', s2.get_age())

# 类方法
Student.run(23)
s1.run(34)

# 静态方法
s2.do('ase')
Student.do('wefr')


class Price:
    def __init__(self, lowprice=0, highprice=0):
        if not Price.isvalid(lowprice, highprice):
            raise ValueError("Low price should not be higher than high price")
        self._low = lowprice
        self._high = highprice

    @staticmethod
    def isvalid(lowprice, highprice):
        return True if lowprice <= highprice else False

    # 定义访问接口
    @property
    def price(self):
        return self._low, self._high

    @price.setter
    def price(self, twoprices):
        """
        似乎Python的setter只能接受一个参数（除了self外）
        所以这里传入了数组
        """
        if Price.isvalid(twoprices[0], twoprices[1]):
            self._low = twoprices[0]
            self._high = twoprices[1]
        else:
            raise ValueError("Low price should not be higher than high price")


p2 = Price(100.0, 120.0)
print(p2.price)
p2.price = (110, 140)
print(p2.price)
