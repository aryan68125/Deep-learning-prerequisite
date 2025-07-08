import numpy as np

"""1. INDEXING AND SLICING"""
# indexing in python list
my_list = [1, 2, 3, 4, 5]
print(f"printing python list ===> \n {my_list}")
print(f"indexing in python list: {my_list[0:3]}")
# python list also supports -1 indexing so we can get access to the very last element
print(f"indexing in python list with -1: {my_list[-1]}")

# indexing in numpy 1D array
np_array = np.array([1, 2, 3, 4, 5])
print(f"printing numpy 1D array ===> \n {np_array}")
print(f"indexing in numpy array: {np_array[0:3]}")
# just like python list it supports reverse index
print(f"indexing in numpy array with -1: {np_array[-1]}")

# indexing in numpy 2D array
np_2d_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"printing numpy 2D array ===> \n {np_2d_array}")
# np_2d_array[row, column]
print(f"select element in numpy 2D array: {np_2d_array[1, 1]}")  # Accessing element at row 0, column 1
"""2. ITERATING THROUGH A NUMPY ARRAY"""
"""3. STACKING TOGEATHER TWO ARRAYS"""
"""4. INDEXING A BOOLEAN NUMPY ARRAY"""