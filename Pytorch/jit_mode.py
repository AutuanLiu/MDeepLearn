#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
   File Name：jit_mode
   Description :  numba jit mode 定义（常用模式）
   Email : autuanliu@163.com
   Date：18-2-1
"""

from numba import jit
# from numba import (int16, uint16, int32,
#                    int64, uint32, uint64,
#                    void, float32, float64)

# 模式的定义
jit_cpu = jit(nopython=True, parallel=True)
jit_gpu = jit(nopython=True, parallel=True, target='gpu')
