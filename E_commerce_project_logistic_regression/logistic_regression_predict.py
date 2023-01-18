import numpy as np
#get the data from the e_commerce_project_process_data.py file by importing it as a module
from e_commerce_project_process_data import get_binary_data

X,Y = get_binary_data()

# get the dimensionality of the dataset
# randomly initialize the weights
# D is the normal standard distribution
D = X.shape[1] # from here we can initialize the weights
# we can use that "dimensionality of the dataset" to initialize the weights of the Logistic regression model
# weights is the one dimensional vector of size D and we will initialize it randomely from the normal standard distribution.
W = np.random.randn(D) # D is the size here
# b is the baised term so that's the scalar
b = 0

# make predictions
#sigmoid function
def sigmoid(a):
    return 1/(1+np.exp(-a))

#this function is responsible for making predictions
def forward(X, W, b):
    return sigmoid(X.dot(W)+b)

# P_Y_given_X = we have n samples and we have D features but now we have outputs and each of those outputs is just a single scalar probability.
# so P_Y_given_X should be an n length one dimensional vector.
# P_Y_given_X does not give predictions as an output per say. So because it is a classifier, a binary classifier , we want the predictions
# to be either zero or one.In order to get those predictions we have to round these probabilities.
P_Y_given_X = forward(X, W, b)
# here we are rounding up the probabilities to get the the predictions as an output.
# So as long as the probability is bigger than 50% we will say that it's a one otherwise it's a zero.
'''
Now there are some rear cases where people change the threshold.
'''
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
