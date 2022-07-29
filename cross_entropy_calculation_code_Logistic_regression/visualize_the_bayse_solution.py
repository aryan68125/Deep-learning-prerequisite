import numpy as np
import matplotlib.pyplot as plt

N=100 # Number of samples
D=2 # Number of dimensions or features

X = np.random.randn(N,D) # matrix of data N by D or samples by features in each samples

# this times we are gonna have labels because we want to calculate the error
#set the first 50 entries of samples to be centered at x=-2 and y=-2
#you can do this by subtracting the matrix of 1s multiplied by 2
X[:50,:]=X[:50,:] - 2*np.ones((50,D)) #centered at -2,-2
X[50:,:]=X[50:,:] + 2*np.ones((50,D)) #centered at +2,+2

#set the targets
#[0]*50 target 0 + [1]*50 target 1
T = np.array([0]*50+[1]*50)

# concatenate a column with 1s or biase
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones,X),axis=1)

def sigmoid(z):
    return 1/(1+np.exp(-z))

#get the closed form solution 0 is the bias here
w = np.array([0,4,4])

# this should give us the line y=-X
# use pyplot scatter plot to plot the X's samples in all rows in 0th and 1st columns
# set the color to targets and size to 100 alpha 2.5 to the dots are transparent
plt.scatter(X[:,0],X[:,1], c=T, s=100, alpha = 0.5)

x_axis = np.linspace(-6,6,100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()
