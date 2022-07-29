import numpy as np

N=100 # Number of samples
D=2 # Number of dimensions or features

X = np.random.randn(N,D) # matrix of data N by D or samples by features in each samples

# this times we are gonna have labels because we want to calculate the error
#set the first 50 entries of samples to be centered at x=-2 and y=-2
#you can do this by subtracting the matrix of 1s multiplied by 2
X[:50,:]=X[:50,:] - 2*np.ones((50,D)) #centered at -2,-2
X[50:,:]=X[50:,:] + 2*np.ones((50,D)) #centered at +2,+2

#set the targets
T = np.array([0]*50+[1]*50)

# concatenate a column with 1s or biase
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones,X),axis=1)

#randomly initialize the weights
w = np.random.randn(D+1)
#calculate the model output
z=Xb.dot(w)

def sigmoid(z):
    return 1/(1+np.exp(-z))

Y=sigmoid(z)

#create a function to calcualte the cross entropy
def cross_entropy(T,Y):
    E=0
    #sum over each individual cross entropy error for each sample
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E-= np.log(1-Y[i])
    return E

print(f"cross entropy error function without close form solution : {cross_entropy(T,Y)}")

#use the closed form solution to logistic regression and see how good that solution is
# NOTE : THIS CLOSE FORM SOLUTION WILL ONLY WORK IF THE DATASET THAT WE ARE WORKING ON IS SMALL AND NOT VERY LARGE BECAUSE IN CLOSE FORM SOLUTION WE HAVE TO TAKE
#         INVERSE OF A MATRIX WHICH IS VERY COMPUTATIONAL HEAVY
# this will work here because we have equal variances in both classes so the variance is 1 which is default for a numpy random normal
# and so the weights only depends on the means
# biase is 0 and weights are 4
w = np.array([0,4,4])
z = Xb.dot(w)
Y = sigmoid(z)
print(f"cross entropy error function with close form solution : {cross_entropy(T,Y)}")
