import numpy as np

# Created a one dimentional array
one_d_arr = np.array([1,2,3,4,5,5])
print(f"one_d_arr ===> {one_d_arr}")
print(f"dimentions of the array ===> {one_d_arr.ndim}")
print(f"shape of the array ===> {one_d_arr.shape}")
print(f"Byte size of the elements in the array ===> {one_d_arr.itemsize}")
print(f"data type of the array ===> {one_d_arr.dtype}")
print(f"size of the array ===> {one_d_arr.size}")
print("\n")

# Create a two dimentional array
two_d_arr = np.array([[1,2,3,4,5],[6,7,8,9,0]])
print(f"two_d_arr ===> {two_d_arr}")
print(f"dimentions of the array ===> {two_d_arr.ndim}")
print(f"Shape of the array ===> {two_d_arr.shape}")
print(f"Byte size of the elements in the array ===> {two_d_arr.itemsize}")
print(f"data type of the array ===> {two_d_arr.dtype}")
print(f"size of the array ===> {two_d_arr.size}")
print("\n")

# If you want to create a np_array with a different datatype 
# initialize the np array with a float datatype
two_d_arr = np.array([[1,2,3,4,5],[6,7,8,9,0]], dtype=np.float64)
print(f"two_d_arr ===> {two_d_arr}")
print(f"dimentions of the array ===> {two_d_arr.ndim}")
print(f"Shape of the array ===> {two_d_arr.shape}")
print(f"Byte size of the elements in the array ===> {two_d_arr.itemsize}")
print(f"data type of the array ===> {two_d_arr.dtype}")
print(f"size of the array ===> {two_d_arr.size}")
print("\n")

# initialize the np array with a complex datatype to store complex numbers
complex_array = np.array([[1,2,3,4,5],[6,7,8,9,0]], dtype=np.complex)
print(f"complex_array ===> {complex_array}")
print("\n")

# How to initialize a np array with some placeholder number
zeros_array = np.zeros((3,4))
print(f"zeros_array ===> {zeros_array}")
print("\n")


# numpy also has a function that is similar to python's range function
# np.arange(1,10,2) here 1 = starting number 10 is the ending number and 2 is the steps i.e the number of steps that the arange must skip when creating a numpy array
arr = np.arange(1,10,2)
print(f"arr ===> {arr}")
print("\n")



# Suppose you want to generate 20 numbers between 1 and 10 that are linearly spaced then you can use linspace function This is pretty useful when you want to create a linear sequence of numbers
arr = np.linspace(1,10,20)
print(f"arr_linspace ===> {arr}")
print("\n")


# reshape array using numpy 
# reshaping means that if you have an array that has the shape of 3 X 2 then you can reshape it to be 2 X 3
original_arr = np.array([[1,2,3],[4,5,6]],dtype=np.float64)
print(f"original_arr ===> \n {original_arr}")
print(f"shape of the original_arr ===> {original_arr.shape}")
reshaped_arr = original_arr.reshape(3,2)
print(f"reshaped_arr ===> \n {reshaped_arr}")
print(f"shape of the reshape_arr ===> {reshaped_arr.shape}")
print("\n")

# If you want to flatten your 2D array into a oneD array then you can use ravel function
# One thing to remember is that ravel() function do not touch the original array so this the one thing you need to keep in mind when working with ravel function
two_d_array = np.array([[1,2,3],[4,5,6]],dtype=np.int64)
print(f"two_d_array ===> \n {two_d_array}")
print(f"two_d_array shape ===> {two_d_array.shape}")
flattened_array = two_d_array.ravel()
print(f"flattened_array ===> \n {flattened_array}")
print(f"flattened_array shape ===> {flattened_array.shape}")
print("\n")

#Some mathematical functions in numpy
two_d_array = np.array([[1,2,3],[4,5,6]],dtype=np.int64)
print(f"array ===> \n {two_d_array}")
# get minimum element in np array
print(f"minimum element in np_array ===> {two_d_array.min()}")
# get maximum element in np array
print(f"maximum element in np_array ===> {two_d_array.max()}")
# get the summation of all the elements in a np array
print(f"summation of all the elements in the array ===> {two_d_array.sum()}")
# There is a concept of axis in numpy
""""
This function is used to prettify the shape of numpy arrays for better readability.
"""
def pretty_shape(name, arr):
    shape = arr.shape
    if len(shape) == 1:
        return f"> {name}.shape\n({shape[0]} columns, )\n"
    elif len(shape) == 2:
        return f"> {name}.shape\n({shape[0]} rows, {shape[1]} columns)\n"
    elif len(shape) == 3:
        return f"> {name}.shape\n({shape[0]} sheets, {shape[1]} rows, {shape[2]} columns)\n"
    else:
        return f"> {name}.shape\n{shape} (unsupported format)\n"
    
vec = np.array([1, 2, 3, 4])

row = np.array([[1, 2, 3, 4]])

col = np.array([[1],
                [2],
                [3],
                [4]])

mat = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8]])

book = np.array([[[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0]],
                 
                 [[1.1, 2.1, 3.1, 4.1],
                  [5.1, 6.1, 7.1, 8.1]],

                 [[1.2, 2.2, 3.2, 4.2],
                  [5.2, 6.2, 7.2, 8.2]]])
for name, arr in [('vec', vec), ('row', row), ('col', col), ('mat', mat), ('book', book)]:
    print(pretty_shape(name, arr))
# for 1D array 
print(f"Calculated the sum of elements along axis 0 (row axis) of a 1D array (row) ===> {row.sum(axis=0)}")
print(f"Calculated the sum of elements along axis 1 (column axis) of a 1D array (row) ===> {row.sum(axis=1)}")
print(f"Calculated the sum of elements along axis 0 (row axis) of a 1D array (col) ===> {col.sum(axis=0)}")
print(f"Calculated the sum of elements along axis 1 (column axis) of a 1D array (col) ===> {col.sum(axis=1)}")
# for 2D array
print(f"Calculated the sum of elements along axis 0 (row axis) of a 2D array ===> {mat.sum(axis=0)}")
print(f"Calculated the sum of elements along axis 1 (column axis) of a 2D array ===> {mat.sum(axis=1)}")
# for 3D array
print(f"Calculated the sum of elements along axis 0 of a 3D array (sheets) ===> {book.sum(axis=0)}")
print(f"Calculated the sum of elements along axis 1 of a 3D array (rows) ===> {book.sum(axis=1)}")
print(f"Calculated the sum of elements along axis 2 of a 3D array (columns) ===> {book.sum(axis=2)}")
print("\n")
# there is a concept of axis in numpy 