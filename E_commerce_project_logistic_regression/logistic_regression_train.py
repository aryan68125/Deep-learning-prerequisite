import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#get the data from the e_commerce_project_process_data.py file by importing it as a module
from e_commerce_project_process_data import get_binary_data

#get binary data from the e_commerce.csv file in the folder
X,Y = get_binary_data()

#shuffle everything in X and Y jsut incase if there are elements inside of it are in order
X,Y = shuffle(X,Y)

# create a train and a test sets
# starts from 0 and ends at 100th position before the last index in the array
Xtrain  = X[:-100]
Ytrain = Y[:-100]

#starts at the 100th position before the last index of the array and ends at the last position of the index of the array
Xtest = X[-100:]
Ytest = Y[-100:]

#randomely initialize the weights again
#get the dimensionality of the dataset
D = X.shape[1] #shape of D to be one
#we can use that "dimensionality of the dataset" to initialize the weights of the Logistic regression model
W = np.random.randn(D) #weights setting up to be at random
# b is the baised term so that's the scalar
b=0 #biase

#sigmoid function
def sigmoid(a):
    return 1/(1+np.exp(-a))

#forward function
#this function is responsible for making predictions
#passing on the X : Data Matrix, W : Weight matrix , b : bias
def forward(X,W,b):
    return sigmoid(X.dot(W) + b)

#function that will give us the classification rate
# it returns the mean of Y==P
def classificationRate(Y,P):
    #This may look like it will return an array of boolean but it actually returns 0s and 1s
    #It will divide the number of correct prediction by total number of predictions
    return np.mean(Y==P)

#corss_entropy function this is gonna take in Targets which is Ytrain and Ytest and PofY
def cross_entropy(T,pY):
    return -np.mean(T*np.log(pY) + (1-T)*np.log(1-pY))

#Now we are gonna enter our main training loop
#array of training cost
train_costs = []
#array of test cost
test_costs = []
#set our learning rate to be 0.001
learning_rate = 0.001
# we are gonna go for 10,000 epox
for i in range(10000):
    #in each iteration calculate pYtrain
    #passing Xtrain through the logistic regression to make predictions on the train dataset
    # P_Y_given_X = pYtrain in training dataset
    pYtrain = forward(Xtrain , W, b)
    #passing Xtest through the logistic regression to make predictions on the test dataset
    # P_Y_given_X = pYtest in test dataset
    pYtest = forward(Xtest, W, b)

    #calculate the training cost
    ctrain = cross_entropy(Ytrain, pYtrain)
    #calculate the test cost
    ctest = cross_entropy(Ytest,pYtest)
    #now append those to the list of cost
    train_costs.append(ctrain) #append ctrain in the train_costs list
    test_costs.append(ctest) #append the ctest in the test_costs list

    #Now we are ready to do the gradient descent
    #so the equation of dradient descent implemented here is the vectorized version of the equaiton of gradient descent that is solved in the NoteBook
    W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain) # defining the weights via gradient descent equation
    b-= learning_rate*(pYtrain-Ytrain).sum()

    # in every 1000 steps print the training cost and the test cost
    if i %1000==0:
        print(f"Steps : {i}")
        print(f"Training cost : {ctrain}")
        print(f"Test cost : {ctest}")

#finally when all that's done we can print the final train classification rate
#here classification rate will be the train predictions
print(f"Final train classification rate : {classificationRate(Ytrain, np.round(pYtrain))}")
#here classification rate will be the test predictions
print(f"Final test classification rate : {classificationRate(Ytest, np.round(pYtest))}")

#plot the train and test costs on the graph
legend1 = plt.plot(train_costs, label = 'train cost')
legend2 = plt.plot(test_costs, label = 'test cost')
plt.legend([legend1, legend2])
plt.show()
