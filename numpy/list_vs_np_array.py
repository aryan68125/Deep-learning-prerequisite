import numpy as np
import time 
import sys

# This is a demo that shows the difference between a numpy array and a python list

# Demo of memory size taken by np array and python list
l = list(range(1000))
print(f"Memory taken by python list ===> {sys.getsizeof(5)*len(l)}")
array = np.arange(1000)
print(f"Memory taken by numpy array ===> {array.size*array.itemsize}")\

# This is a demo of how numpy forces different data-types to be converted into the same data-type
# since here exist a string in the numpy array converts every element in the array to string <U32
arr = np.array([1, 2.5, True, "hello"])
print(arr)
print(arr.dtype)

# demo to show that the numpy array is faster than a python list
size = 9999999
list_1 = list(range(size))
list_2 = list(range(size))
np_array_1 = np.arange(size)
np_array_2 = np.arange(size)
# measure the time between the list processing and numpy array processing
# measure the time for the python list
start = time.time()
list_result = [x+y for x,y in zip(list_1,list_2)]
end = time.time()
print(f"Time taken to process the list ===> {(end-start)*1000}")
# print(f"result list ===> {list_result}")
# measure the time for the numpy array list
start2 = time.time()
result_np = np_array_1 + np_array_2
end2 = time.time()
print(f"Time taken to process the np_array ===> {(end2-start2)*1000}")
# print(f"result np_array ===> {result_np}")


