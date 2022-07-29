import numpy as np
import pandas as pd

def get_data():
    # read the data from the the file system which is in form of csv and convert it to a dataframe
    df = pd.read_csv('ecommerce_data.csv')
    # convert that dataframe into a numpy matrix
    #data = df.as_matrix() if this doen't work then
    #Replacing .as_matrix() with .values() also resulted in an error, but replacing it with .to_numpy() worked perfectly
    data = df.to_numpy()

    # split the data into x and y
    # x here is the input and y here is the output or the target
    X = data[:, :-1] #every thing upto the last column and all the rows because y is our last column
    Y = data[:,-1] # only the the last column and all the rows

    # normalize the numerical columns
    # x1 = (x1 - mean of x1)/ standard diviation of x1
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std() # X1 = is_mobile if the customer is using a mobile version of the website
    # x2 = (x2 - mean of x2)/ standard diviation of x2
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std() # X2 = n_producs viewed by the customer

    # work on the categorical column
    # time of the day
    N,D = X.shape # get the shape of the original X
    # make a new X of shape N by D+3 because there are 4 different categorical values
    X2 = np.zeros((N,D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)]

    #on-hot encoding for the other four columns use range in python3 instead of xrange
    for n in range(N):
        #get the time of day
        #remember this is either 0,1,2 or 3
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1
    # another method of one hot encoding is to create a new matrix
    #create a new matrix
    Z = np.zeros((N,4)) # size N and 4 for the 4 columns
    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    # X2[:,-4:] = Z
    assert(np.abs(X2[:,-4:]-Z).sum()<10e-10)
    return X2,Y

# for the logistic class we only want the binary data we don't want the full dataset
def get_binary_data():
    X,Y = get_data()
    #this is gonna filter it by only taking classes 0 and 1
    X2 = X[Y<=1] #X2 is all the Xs for where Y is either 0 or 1
    Y2 = Y[Y<=1] #Y2 is Ys where Y is less than or equal to 1
    return X2,Y2
