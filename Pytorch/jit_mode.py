#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
   File Name：jit_mode
   Description :  numba jit mode 定义（常用模式）
   pytorch 中也自带 jit 编译
   Email : autuanliu@163.com
   Date：18-2-1
"""

from numba import float32, float64, generated_jit, int16, int32, int64, jit, vectorize, void

# jit 模式的定义
jit_cpu = jit(nopython=True, parallel=True)
jit_gpu = jit(nopython=True, parallel=True, target='cuda')

# generated_jit 模式的定义
gjit_cpu = generated_jit(nopython=True, parallel=True)
gjit_gpu = generated_jit(nopython=True, parallel=True, target='cuda')

# ufunc example
vec_cpu = vectorize([
    void(int64, int64),
    int16(int16, int16),
    int32(int32, int32),
    int64(int64, int64),
    float32(float32, float32),
    float64(float64, float64)
])

vec_gpu = vectorize(
    [
        void(int64, int64),
        int16(int16, int16),
        int32(int32, int32),
        int64(int64, int64),
        float32(float32, float32),
        float64(float64, float64)
    ],
    target='cuda')
