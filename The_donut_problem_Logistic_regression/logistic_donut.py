import numpy as np
import matplotlib.pyplot as plt

#CREATING DATA FOR DEMOSTRATION OF THE DONUT PROBLEM
#in this example we are gonna use a lot more data points so that we can see something substantial
N=1000 # number of samples
D = 2 # number of features or dimensions

# so here we have two radiuses
#we have inner radius
R_inner = 5
# we have outer radius
R_outer = 10

#INNER RADIUS
# we are gonna set a uniformly distributed variable for half the data that depends on the inner radius so it's spread around 5
R1 = np.random.randn(int(N/2)) + R_inner
# we are gonna generate some angles so these are polar coordinates that are uniformly distributed
# formula used to find theta here is 2piR here R is N/2 which is half of the data randomly selected by numpy
theta = 2*np.pi*np.random.random(int(N/2)) # polar coordinates
#here we are converting the polar coordinates into xy coordinates
# {cos(Theta) * R and sin(Theta) * R} tranpose this entire matrix that goes along the rows
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

#OUTER RADIUS
# we are gonna do the samething for the outer radius
# we are gonna set a uniformly distributed variable for half the data that depends on the outer radius so it's spread around 10
R2 = np.random.randn(int(N/2)) + R_outer
# we are gonna generate some angles so these are polar coordinates that are uniformly distributed
# formula used to find theta here is 2piR here R is N/2 which is half of the data randomly selected by numpy
theta = 2*np.pi*np.random.random(int(N/2)) # polar coordinates
#here we are converting the polar coordinates into xy coordinates
# {cos(Theta) * R and sin(Theta) * R} tranpose this entire matrix that goes along the rows
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

# here we are gonna calculate our entire X by concatenating X_inner and X_outer arrays into one
X = np.concatenate([X_inner , X_outer]) # X input data matrix
#setting up the targets array
#we say the first set is 0 and the second set is 1 for half the data in N (Number of smaples) i.e for 500 samples
T = np.array([0]*(int(N/2)) +[1]*(int(N/2))) #Targets

#Now plotting these data points so we can have a look how these look like
#X[:,0] selecting all the rows in x matrix and selecting 0th column in the X matrix
#X[:,1] selecting all the rows in the x matrix and selecting the 1st column in the X matrix
plt.scatter(X[:,0], X[:,1], c=T)
plt.show()

# create a comlumn of 1s for the biased term
ones = np.array([[1]*N]).T

'''
TRICK OF THE DONUT PROBLEM:
       The trick with the donut problem is -> We are going to create yet another column which represents the radius of a point
       this will make your data points linearly separable
'''
r = np.zeros((N,1)) # creating a N size and 1 dimension of numpy array for radius
#manually calculate the radiuses
for i in range(N):
    r[i] = np.sqrt(X[i,:].dot(X[i,:]))
#Now then I do my concatenation the ones and the radiuses and It's all togeather
#axis=1 -> adding ones npArray, r npArray and X npArray by column
Xb = np.concatenate((ones,r,X), axis = 1)
#Randomely initialize the weights
w = np.random.rand(D+2)

z = Xb.dot(w)

def sigmoid(z):
    return 1/(1+np.exp(-z))

#calculating the output
Y = sigmoid(z)

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

#set the learning rate to 0.0001
learning_rate = 0.0001

#array to hold the errors
error = []
#running for loop for 5000 epox
'''
In general you will have to experiment a little bit to find the right number for these values or you could use something like cross-validation
'''
for i in range(5000):
    #keep track of cross entropy so we can see how it evolves over time
    e = cross_entropy(T,Y)
    error.append(e)
    #print the cross_entropy every 100 times
    if i % 100 == 0:
        print(f"Cross_entropy error function : {e}")

    # using gradient with L2 regularization to find the optimal weights for the entropy function
    # L2 penalty here is 0.01 and L2 regularization is 0.01*w
    w+=learning_rate * (np.dot((T-Y).T, Xb) - 0.01*w)
    #re-calculating the output (Predicted value by the model)
    Y = sigmoid(Xb.dot(w))

#plot the entropy error as it evolve over time in the pyplot graph
plt.plot(error)
plt.title("Cross-entropy")
plt.show()

print(f"Final weights : {w}")
#when we are classifying we are actually rounding the output from the sigmoid function here in this case it's Y
print(f"Final classification rate : {1-np.abs(T-np.round(Y)).sum() / N}")

'''
    bias             radius        x coordinate    y coordinate
[-1.19218610e+01  1.60701810e+00  1.07760033e-03  9.60229312e-03]
The output of this is that our x and y are pretty close to zero
Classification doesn't really depend on the x and y coordinate at all is what this model has found
This model has found that the classification depends on the bias
here the radius that we'v put in the small radius and we have automatically have this bias to be -ve and that pushes the classification
towards zero
If the radius is bigger then it pushes the classification towards one
So that's how you can solve the donut problem
'''
