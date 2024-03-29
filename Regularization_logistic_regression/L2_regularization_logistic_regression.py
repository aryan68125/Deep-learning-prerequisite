import numpy as np

N=100 # Number of samples
D=2 # Number of dimensions or features

X = np.random.randn(N,D) # matrix of data N by D or samples by features in each samples

#setting up the gaussian data set
# this times we are gonna have labels because we want to calculate the error
#set the first 50 entries of samples to be centered at x=-2 and y=-2
#you can do this by subtracting the matrix of 1s multiplied by 2
X[:50,:]=X[:50,:] - 2*np.ones((50,D)) #centered at -2,-2
X[50:,:]=X[50:,:] + 2*np.ones((50,D)) #centered at +2,+2

#set the targets T
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

#Output
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

print(f"cross entropy error function without gradient descent : {cross_entropy(T,Y)}")

#selecting weight via gradient descent to improve the performance of our logistic regression model
#step size == learning rate here learning rate is 0.1
learning_rate = 0.1

#during gradient descent it has a tendency to find the weight [0,infinity,infinity] which cannot be calculated by a computer that's why we are doing
#regularization to prevent over fitting and to prevent gradient descent from setting the higher number weights
#L2 penalty lambda constant
L2=0.1
# performing 100 iteration of gradient descent
for i in range(10000):
    #printing the cross_entropy after every 10 steps so see wheather it's decreasing or not
    if (i%10==0):
        print(f"cross entropy error function with gradient descent : {cross_entropy(T,Y)}")

    #finding weight using gradient descent
    #for now we are doing it in the for loop later simple use numpy to [ np.dot((T-Y).T, Xb) -> X.T.dot(T-Y) ] it's much faster than a python for loop also it will calculate it all at once
    #Regularization component is [ (Lambda constant = 0.1(L2) = l2 penalty)*weight which is w here ]
    #since ith weight in the left side depends on the ith weight on the right side we can do it in the vector form
    w+=learning_rate*(np.dot((T-Y).T, Xb)-L2*w)
    #recalculate the output Y here
    Y = sigmoid(Xb.dot(w))
print(f"Final weight for the logistic regression is : {w}")
