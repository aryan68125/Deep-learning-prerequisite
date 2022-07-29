# Demonstration of L1 regularization
# The strategy is to generate some data where the input is a fat matrix and Y will only depend only on the few of the features the rest of them will just be noise
# Then we will use L1 regularization to see if we can find the sparse set of weights that identifies the useful dimensions of X

import numpy as np
import matplotlib.pyplot as plt

print("L1 Regularization")

# sigmoid function to calculate the out put of the logistic regression function
# sigmoid = 1/1+e^-x
def sigmoid(z):
    return 1/(1+np.exp(-z))

N = 50 # number if samples
D = 50 # number of features per sample

# set the x matrix to have uniformly distributed numbers between +5 and  -5
# here we are subtracting 0.5 percent from np.random.radnom((N,D)) so that we can ceter this matrix datapoints around 0
X = (np.random.random((N,D)) -0.5)*10

# The true weights of this is [1,0.5,-0.5] so only the first 3 dimensions only effect the output and the rest are 0
# So the last 47 dimensions do not effect the output at all
# When we say dimensions we are reffering to the features that each sample has in the X matrix or the input matrix
true_w = np.array([1,0.5,-0.5]+[0]*(D-3))

#Generating Y the targets
#np.round( [sigmoid(X.dot(true_w)) = gives us the output predictions] + [np.random.randn(N)*0.5 =  random noise] )
# NOTE: LATER WE WILL HAVE TO CHANGE THE NOISE INTRODUCED HERE
Y = np.round(sigmoid(X.dot(true_w) + np.random.randn(N)*0.5))

#Perform gradient descent
#define a cost array to keep track of the cost
costs = []
'''
Randomely initialize the weights
you want your numbers to be inside a small range, since the sigmoid is flat at extreme values.
i.e. we want variance = 1 which is why we normalized the data first.
So if we have y = w1x1 + ... wDxD, then var(y) = var(w1)var(x1) + ... var(wD)var(xD)
If we set var(wi) = 1/D, then we achieve var(y) = 1, since var(xi) = 1 by normalization.
'''
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
#L1 penalty changing l1 penalty 2.0 -> 10.0
l1 = 1.0
# for loop to train the model for 10000 epox
for t in range(10000):
    # find Y hat
    Yhat = sigmoid(X.dot(w))
    delta = Yhat - Y
    #gradient descent
    w = w -learning_rate*(X.T.dot(delta) + l1*np.sign(w))

    #find and store the cost
    #THIS WILL BE SLIGHTLY MODIFIED LATER
    cost = -(Y*np.log(Yhat) + (1-Y)*np.log(1-Yhat)).mean() + l1*np.abs(w).mean()
    costs.append(cost)

#plot the cost in the graph via pyplot matplotlib
plt.plot(costs)
plt.show()

# plot the true w vs the w we found so that we can comare them
plt.plot(true_w,label = 'true w')
plt.plot(w,label = 'w map')
plt.legend()
plt.show()
