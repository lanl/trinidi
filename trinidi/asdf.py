"""Some asdf functions."""

import numpy as np


def fibs(N):
    r"""Generate a list of the first N Fibonacci numbers.

    The Fibonacci sequence is defined as

    .. math::
        f(n+2) = f(n+1) + f(n)\;,

    where :math:`f(1) = f(2) = 0`.
    """
    fib = lambda n: n if n < 2 else fib(n - 1) + fib(n - 2)

    return np.array([fib(i) for i in range(N)])
