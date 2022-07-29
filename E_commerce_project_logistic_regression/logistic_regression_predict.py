import numpy as np
#get the data from the e_commerce_project_process_data.py file by importing it as a module
from e_commerce_project_process_data import get_binary_data

X,Y = get_binary_data()

#get the dimensionality of the dataset
D = X.shape[1]
#we can use that "dimensionality of the dataset" to initialize the weights of the Logistic regression model
W = np.random.randn(D) # D is the size here
# b is the baised term so that's the scalar
b = 0

#sigmoid function
def sigmoid(a):
    return 1/(1+np.exp(-a))

#this function is responsible for making predictions
def forward(X, W, b):
    return sigmoid(X.dot(W)+b)

P_Y_given_X = forward(X, W, b)
predictions = np.round(P_Y_given_X)

#It's gonna take in targets = Y and predictions = P
def classification_rate(Y,P):
    #This may look like it will return an array of boolean but it actually returns 0s and 1s
    #It will divide the number of correct prediction by total number of predictions
    return np.mean(Y==P)

#print the score
print(f"Score : {classification_rate(Y,predictions)}")

# if we choose more randomely we are not gonna do that well
# to get the score more accurately we need to train these weights for this we will use gradient descent
