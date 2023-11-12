# 二进制快速幂
from numba import jit, int32
import time

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

@jit(int32(int32, int32),nopython=True, cache=True)
def fast_pow(base: int, exp: int):
    ans = 1
    while exp > 0:
        if exp & 1:
            ans = ans * base % MOD
        base = base * base % MOD
        exp >>= 1
    return ans


def my_pow(a, n):
    return pow(a, n,MOD)

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

if __name__ == '__main__':
    bit_wise_cal()
