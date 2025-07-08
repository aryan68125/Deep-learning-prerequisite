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
# slicing in numpy 2D array
# np_2d_array[row,column] here select row = 0:2 column = 2 hence the code will select 3 and 6 since these elements are in 0th and 1st row and in column 2
# when slicing a 2D array like this 0:2 the code doesn't include the number 2 which means it will select the element from 0th to 1st row this is the same when you slice the columns as well
print(f"slice a 2D array in a way to select the 2nd element in row 0 and row1 : {np_2d_array[0:3,2]}")
print(f"slicing a 2D array to select the first two column from the first row : {np_2d_array[0,1:3]}")
# np_2d_array[rows,columns] when you don't provide rows like this ':' but you don't provide any number to slice the rows then in this case all the rows will be selected and in case of coumns we have passed 0:2 which means the 0th column and the 1st column will be selected
print(f"slicing a 2D array to select first two columns from all the rows : \n {np_2d_array[:,0:2]}")
# you can also do reverse indexing in numpy 2D array
print(f"reverse indexing in numpy 2D array: {np_2d_array[-1,-1]}")  # Accessing the last element in the last row
# if you provie the reverse indexing like this then it will select the last row from the 2D array
print(f"reverse indexing to select last row from the 2D array: {np_2d_array[-1]}")
# flatten a 2D array into a 1D array 
print(f"flatten a 2D array into a 1D array: {np_2d_array.flatten()}")
"""2. ITERATING THROUGH A NUMPY ARRAY"""
"""3. STACKING TOGEATHER TWO ARRAYS"""
"""4. INDEXING A BOOLEAN NUMPY ARRAY"""