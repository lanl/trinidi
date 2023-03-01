

import numpy as np

def fibs(N):

    fib = lambda n: n if n < 2 else fib(n-1) + fib(n-2)
    
    return np.array([fib(i) for i in range(N)])




