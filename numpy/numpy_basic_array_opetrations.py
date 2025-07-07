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
two_d_array = np.array([[1,2,3],[4,5,6]],dtype=np.int64)
print(f"two_d_array ===> \n {two_d_array}")
print(f"two_d_array shape ===> {two_d_array.shape}")
flattened_array = two_d_array.ravel()
print(f"flattened_array ===> \n {flattened_array}")
print(f"flattened_array shape ===> {flattened_array.shape}")
