import numpy as np
import pandas as pd

def get_data():
    # read the data from the the file system which is in form of csv and convert it to a dataframe
    df = pd.read_csv('ecommerce_data.csv')
    # convert that dataframe into a numpy matrix
    #data = df.as_matrix() if this doen't work then
    #Replacing .as_matrix() with .values() also resulted in an error, but replacing it with .to_numpy() worked perfectly
    data = df.to_numpy()
    #shuffle the data
    np.random.shuffle(data)

    # split the data into x and y
    # x here is the input and y here is the output or the target
    X = data[:, :-1] #every thing except the last column and all the rows because y is our last column which is our target or the output
    Y = data[:,-1] # only the the last column and all the rows as last column is y which is our target or the output

    # normalize the numerical columns
    # x1 = (x1 - mean of x1)/ standard diviation of x1
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std() # X1 = is_mobile if the customer is using a mobile version of the website
    # x2 = (x2 - mean of x2)/ standard diviation of x2
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std() # X2 = n_producs viewed by the customer

    # work on the categorical column
    # ONE-HOT ENCODING
    # time of the day
    N,D = X.shape # get the shape of the original X
    # make a new X of shape N by D+3 because there are 4 different categorical values
    # why we are using D+3? -> It is because that we have 4 categories and if we want to encode that we need 4 new columns
    # but we can also replace the existing column which is why we only need to add three more columns.
    X2 = np.zeros((N,D+3))
    # first thing is that we will copy all the non categorical data first
    X2[:,0:(D-1)] = X[:,0:(D-1)] # non categorical columns
    #on-hot encoding for the other four columns use range in python3 instead of xrange
    for n in range(N):
        #first get the actual encoded value that's encoded as an integer.
        #get the time of day
        #remember this is either 0,1,2 or 3
        t = int(X[n,D-1])
        #assign t in its appropriate place in X2 numpy array
        # so Why is it t+D-1? -> SO D-1 will give us the final column in the original matrix which i guess is the 4th last column of our new matrix.
        # and then we just adding t to that. and one of those particular cell in the new matrix should be set to 1 based on what the t is.
        X2[n,t+D-1] = 1
    # another method of one hot encoding is to create a new matrix
    #create a new matrix
    Z = np.zeros((N,4)) # size N and 4 for the 4 columns
    # SO what we are doing here is what we are doing is that we are passing in a set of tuples into the rows and columns.
    # So it's basically a z[(r1,r2,r3,...), (c1,c2,c3,...)] = value
    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    # X2[:,-4:] = Z
    assert(np.abs(X2[:,-4:]-Z).sum()<10e-10)
    return X2,Y

# for the logistic class we only want the binary data we don't want the full dataset
def get_binary_data():
    X,Y = get_data()
    #this is gonna filter it by only taking classes 0 and 1
    # Y<=1 this compares Y which is n length one D array to some number 1.  here it will return a boolean array
    # So X[Y<=1] here Y<=1 works as an index in the X n length array. So here we are gonna accept the indexes that are true but ignore that are false
    # and the same thing we are doing it here when we type Y2 = Y[Y<=1]
    X2 = X[Y<=1] #X2 is all the Xs for where Y is either 0 or 1
    Y2 = Y[Y<=1] #Y2 is Ys where Y is less than or equal to 1
    return X2,Y2
