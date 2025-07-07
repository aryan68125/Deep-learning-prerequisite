import numpy as np
import time 
import sys

# This is a demo that shows the difference between a numpy array and a python list

l = list(range(1000))
print(f"Memory taken by python list ===> {sys.getsizeof(5)*len(l)}")

array = np.arange(1000)
print(f"Memory taken by numpy array ===> {array.size*array.itemsize}")\

# This is a demo of how numpy forces different data-types to be converted into the same data-type
# since here exist a string in the numpy array converts every element in the array to string <U32
arr = np.array([1, 2.5, True, "hello"])
print(arr)
print(arr.dtype)
