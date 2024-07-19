import numba
from numba import njit, types
from numba.typed import Dict

@njit
def inner_function(x):
    return x * x

@njit
def outer_function(f, y):
    return f(y)

# Example usage
result = outer_function(inner_function, 10)
print(result)  # Output: 100