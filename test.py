# 二进制快速幂
import numba
# import torch
import time
import math

import numpy

MOD = 1000000007
REPEAT = 10000


def timeit(times):
    def repeat(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            for _ in range(times):
                ans = func(*args, **kwargs)
            end = time.time()
            print(f"time used:{end - start} s")
            return ans
        return wrapper
    return repeat


# @jit(int32(int32, int32), nopython=True, cache=True)
def fast_pow(base: int, exp: int):
    ans = 1
    while exp > 0:
        if exp & 1:
            ans = ans * base % MOD
        base = base * base % MOD
        exp >>= 1
    return ans


def my_pow(a, n):
    return pow(a, n, MOD)


def bit_wise_cal():
    size = 100000000  # 100 million
    data = [0x2] * size  # Initialize a list with 0x2

    # Record start time
    start = time.time()

    # Perform a bitwise AND operation
    for i in range(size):
        data[i] &= 0x1

    # Record end time
    end = time.time()

    # Calculate elapsed time
    elapsed = end - start
    print(f"Elapsed time: {elapsed} seconds.")


def test_shape():
    a = torch.randn(10000)
    N = 10000
    t1 = time.perf_counter()
    for i in range(N):
        m = len(a)
    t2 = time.perf_counter()
    for i in range(N):
        n = a.shape[0]
    t3 = time.perf_counter()
    print(t2 - t1)
    print(t3 - t2)


def test_eq():
    a = torch.randn(1000)
    b = torch.randn(1000)
    N = 1000
    t1 = time.perf_counter()
    for i in range(N):
        m = (a == b).sum()
    t2 = time.perf_counter()
    for i in range(N):
        n = sum(a == b)
    t3 = time.perf_counter()
    print(t2 - t1)
    print(t3 - t2)


def test_mean():
    a = torch.randn(1000)
    b = torch.randn(1000)
    N = 10000
    t1 = time.perf_counter()
    for i in range(N):
        m = (a == b).sum() / a.shape[0]
    t2 = time.perf_counter()
    for i in range(N):
        n = (a == b).float().mean()
    t3 = time.perf_counter()
    print(t2 - t1)
    print(t3 - t2)


def test_sort():
    N = 10000
    t1 = time.perf_counter()
    for i in range(N):
        a = torch.randn(1000).tolist()
        m = a.sort()
    t2 = time.perf_counter()
    for i in range(N):
        a = torch.randn(1000).tolist()
        n = sorted(a)
    t3 = time.perf_counter()
    print(t2 - t1)
    print(t3 - t2)  # 几乎无差别


def test_bit():
    N = 1_000_000
    t1 = time.perf_counter()
    a = 0b10001000_00000000_01001011_00100101
    _quo = 2 ** 12
    for i in range(N):
        b = a >> 12
    t2 = time.perf_counter()
    for i in range(N):
        b = a // _quo
    t3 = time.perf_counter()
    print(t2 - t1)
    print(t3 - t2)  # 几乎无差别

# @numba.njit(nopython=True)
def test_numba(number):
    # t1 = time.perf_counter()
    sqrt_number = int(numpy.sqrt(number))+ 1
    for i in range(2, sqrt_number):
        if number % i  == 0:
            return False
    # t2 = time.perf_counter()
    # print(t2 - t1)
    return True



    


if __name__ == '__main__':
    test_numba(10000019)
