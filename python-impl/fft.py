"""
参考资料：
https://zhuanlan.zhihu.com/p/422925896
https://blog.csdn.net/weixin_40215443/article/details/94617053
https://www.bilibili.com/video/BV1Ka411m7rz?share_source=copy_web&vd_source=642aa5ba88fa3e18be68d55cd933a7c3
"""

import cmath
import math
from typing import List, Callable, Union


def _reverse_bits(n: int, n_bits: int) -> int:
    """
    二进制位翻转
    例：
    x = 0b001  
    print(bin(_reverse_bits(x, 3)))  
    输出0b100  
    :param n: 待翻转的数据
    :param n_bits: 翻转的操作范围
    :return: 翻转后的数据
    """
    return int(bin(n)[2:].rjust(n_bits, "0")[::-1], base=2)


""""another implementation for reverse bits"""
# def _reverse_bits(n: int, n_bits: int) -> int:
#     n = [(n >> i) & 0x01 for i in range(n_bits)]
#     try:
#         from functools import reduce
#         fn = lambda acc, i_x: (i_x[1] << (n_bits - i_x[0] - 1)) | acc
#         return reduce(fn, enumerate(n), 0)
#     except ImportError:
#         ret = 0
#         for i, x in enumerate(n):
#             ret |= x << (n_bits - i - 1)
#         return ret


def _array_reorder(arr: List) -> List:
    n_bits = math.log2(len(arr))
    assert n_bits - int(n_bits) == 0, "len of the array must be 2 ** n."
    n_bits = int(n_bits)
    indexes = [_reverse_bits(x, n_bits) for x in list(range(len(arr)))]
    return [arr[idx] for idx in indexes]


def get_freqs(n: int, fs: int) -> List[float]:
    """
    获取频率
    :param n: 序列长度
    :param fs: 采样频率
    :return: 频率
    """
    df = fs / (n - 1)  # 分辨率
    return [x * df for x in range(n)]


def fft(data: List[complex], postprocess: Callable = lambda x: abs(x)) -> List[Union[float, complex]]:
    """
    快速傅里叶变换 注意：采样频率要大于信号频率的两倍
    :param data: 时域信号
    :param postprocess: 对于变换后数据后处理的函数
    :return: 变换后的频域信号
    """
    data = _array_reorder(data)
    n_data = len(data)
    log2n = 0 if len(data) == 0 else int(math.log2(n_data))
    for i in range(1, log2n + 1):
        m = 1 << i
        deltawn = cmath.rect(1, -2 * math.pi / m)
        for k in range(0, n_data, m):
            wn = cmath.rect(1, 0)
            for j in range(0, int(m / 2)):
                t = data[k + j + int(m / 2)] * wn
                u = data[k + j]
                data[k + j] = u + t
                data[k + j + int(m / 2)] = u - t
                wn *= deltawn
    return [postprocess(x) for x in data]


def ifft(data: List[complex], postprocess: Callable = lambda x: x.real) -> List[Union[float, complex]]:
    """傅里叶逆变换"""
    data = _array_reorder(data)
    n_data = len(data)
    log2n = 0 if len(data) == 0 else int(math.log2(n_data))
    for i in range(1, log2n + 1):
        m = 1 << i
        deltawn = cmath.rect(1, 2 * math.pi / m)
        for k in range(0, n_data, m):
            wn = cmath.rect(1, 0)
            for j in range(0, int(m / 2)):
                t = data[k + j + int(m / 2)] * wn
                u = data[k + j]
                data[k + j] = u + t
                data[k + j + int(m / 2)] = u - t
                wn *= deltawn
    data = [x / n_data for x in data]
    return [postprocess(x) for x in data]


def band_pass_filter(seq: List[float], f_low: float, f_high: float, fs: int) -> List[float]:
    """
    带通滤波器
    :param seq: 时序序列
    :param f_low: 带通下限
    :param f_high: 带通上限
    :param fs: 采样频率
    :return: 滤波后序列
    """
    len_seq = len(seq)
    freqs = get_freqs(len_seq, fs)
    y_fft = fft(seq, postprocess=lambda x: x)
    for i in range(len_seq // 2, len_seq):
        y_fft[i] = 0j
    for i, f in enumerate(freqs):
        if not (f_low <= f <= f_high):
            y_fft[i] = 0j
    y_ifft = ifft(y_fft, postprocess=lambda x: x.real)
    return y_ifft


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np


    def single_generator(a, fc, phase):
        def single_fn(x):
            return [a * math.sin(2 * math.pi * fc * x_ + phase) for x_ in x]

        return single_fn


    def fft_test():
        n = 2 << 9  # 1024 采样数量1024
        # n = 2 << 7  # 256
        single_fn0 = single_generator(a=10., fc=2, phase=0)
        single_fn1 = single_generator(a=1., fc=50, phase=0.)
        t = list(np.linspace(0, 2, n))
        # y = single_fn0(t)
        y = [x1 + x2 for x1, x2 in zip(single_fn0(t), single_fn1(t))]  # 生成频率2和50的叠加信号
        plt.plot(t, y)
        plt.show()

        # 傅里叶变换
        freqs = get_freqs(n, 1 / (t[1] - t[0]))
        y_fft = fft(y, postprocess=lambda x: x)
        plt.plot(freqs[1:n // 2], [abs(x) for x in y_fft[1: n // 2]])  # 由于傅里叶变换对称 只显示一半
        plt.show()

        # 傅里叶逆变换
        # for i in range(n // 2, n):
        #     y_fft[i] = 0j

        y_ifft = ifft(y_fft)
        plt.plot(t, y_ifft)
        plt.show()

        # 滤波+傅里叶逆变换
        # 将对称的另一半频谱全部置0，并不影响傅里叶变换，如果不置0会影响后续抹除频谱的操作（因为频谱是对称的，如果不置0的话要抹除两次）
        for i in range(n // 2, n):
            y_fft[i] = 0j

        # 抹除待消去的频率
        for i, f in enumerate(freqs):
            if 45. <= f <= 55.:
                y_fft[i] = 0j

        # 逆变换
        y_ifft = ifft(y_fft)
        plt.plot(t, y_ifft)
        plt.show()


    def band_pass_filter_test():
        n = 2 << 9  # 1024
        # n = 2 << 7  # 256
        single_fn0 = single_generator(a=10., fc=2, phase=0)
        single_fn1 = single_generator(a=1., fc=50, phase=0.)
        t = list(np.linspace(0, 2, n))
        # y = single_fn0(t)
        y = [x1 + x2 for x1, x2 in zip(single_fn0(t), single_fn1(t))]  # 生成频率2和50的叠加信号
        plt.plot(t, y)
        plt.show()

        y_filted = band_pass_filter(y, 1, 10, 1 / (t[1] - t[0]))
        plt.plot(t, y_filted)
        plt.show()


    fft_test()
    # band_pass_filter_test()
