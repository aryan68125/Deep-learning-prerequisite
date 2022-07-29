import numpy as np
import matplotlib.pyplot as plt

#Unlike our last donut problem there are only 4 data points

N = 4 #number of samples
D = 2 #Dimensions or features per sample

#XOr is a logic gate It's kind of like Or except that (true ,Xor true) gives you false
#So in XOr it anly allows one thing to be true in order to result in true
#So these are data points which represents different data points with different values of true and false
X = np.array( [ [0,0],[0,1],[1,0],[1,1], ] )

#and in our targets you can write mannually false, true,true,false
T = np.array([0,1,1,0])

#Add the column of 1s to our input data
# create a comlumn of 1s for the biased term
ones = np.array([[1]*N]).T

#plot the data points to see what it looks like
#X[:,0] selecting all the rows in x matrix and selecting 0th column in the X matrix
#X[:,1] selecting all the rows in the x matrix and selecting the 1st column in the X matrix
plt.scatter(X[:,0],X[:,1],c=T)
plt.show()

'''
REASON WHY LOGISTIC REGRESSION ISN'T APPLICABLE IN THIS XOR CASE ->
Here you can see that there are 4 data points 2 are from class yellow and 2 are from class purple.
So the trouble using logistic regression here is that you can't really find the line that will give you a perfect classification
'''

'''
TRICK TO SOLVE XOR PROBLEM VIA LOGISTIC REGRESSION ->
So the Trick is with the XOr problem is we're again going to add another dimension to our input
So We are going to turn it into a 3D peoblem instead of a 2D problem and then we can draw a plane between the two datasets
--> If we multiply the x and y to a new variable we can make the data linearly separable
'''
xy = np.matrix(X[:,0]*X[:,1]).T
#Now then I do my concatenation the ones and the xy npArray and It's all togeather
#axis=1 -> adding ones npArray, xy npArray and X npArray by column
Xb = np.array(np.concatenate((ones,xy,X), axis=1))

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
learning_rate = 0.001

#array to hold the errors
error = []
#running for loop for 5000 epox
'''
In general you will have to experiment a little bit to find the right number for these values or you could use something like cross-validation
'''
for i in range(5000):
    #keep track of cross entropy error so we can see how it evolves over time
    e = cross_entropy(T,Y)
    error.append(e)
    #print the cross_entropy every 100 times
    if i % 100 == 0:
        print(f"Cross_entropy error function : {e}")

    # using gradient with L2 regularization to find the optimal weights for the entropy function
    # L2 penalty here is 0.01 and L2 regularization is 0.01*w
    w+=learning_rate * (np.dot((T-Y).T, Xb) - 0.01*w)
    #re-calculating the output
    Y = sigmoid(Xb.dot(w))

#plot the entropy error as it evolve over time in the pyplot graph
plt.plot(error)
plt.title("Cross-entropy")
plt.show()

print(f"Final weights : {w}")
#when we are classifying we are actually rounding the output from the sigmoid function here in this case it's Y
print(f"Final classification rate : {1-np.abs(T-np.round(Y)).sum() / N}")

'''
These past two example bring out a really interesting point ->
You saw that we could apply the logistic regression to some compex problems by manual feature engineering
So we looked at the data and we determine some features that we could calculate the inputs that would help us improve our classification
Now in machine learning ideally the machine would be able to learn these things and so that is precisely what neural networks do
So in the future we will apply these problems and implement a neural networks that will automatically learn the features like this
'''
