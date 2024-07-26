from numba import njit
from numba.typed import List
@njit
def foo(b):
  print(len(b))
  b[0].append(4) 
  index = b[1].pop()

  print(b)

a = [[0], [0], [0]]




typed_a = List(List(x) for x in a)  # make a numba list of numba lists.
foo(typed_a)