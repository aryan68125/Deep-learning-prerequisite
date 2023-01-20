import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import getBinaryData, sigmoid_cost, sigmoid, error_rate
'''
facial expression recognition problem using logistic regression
code the Logisitc regression model in form of a class so we can use the object of the model
similar to the sciKitLearn library
'''
class LogisticModel(object):
    def __init__(self):
        pass
    #fit function which will train our model
    def fit(self, X,Y,learning_rate = 10e-7, reg=0,epochs=120000, show_fig=False):
        #shuffle X and Y
        X,Y = shuffle(X, Y)
        #split X and y in taining and validation sets so we are going to use another set of data to plot the cost
        Xvalid,Yvalid = X[-1000:],Y[-1000:]
        # Set the X and Y to the rest of X and Y
        X, Y = X[:-1000],Y[:-1000]
        N,D = X.shape
        #initialize the weights
        '''
         Why are we dividing the weights during initialization by the square root of input dimensionality
         self.W = np.random.randn(D)/np.sqrt(D)
         As you know poor weight initialization may or may not lead to poor convergence of your loss per interation. Sometimes your loss
         may even explode to infinity.
         NOTE : It will not work all the time also in some cases it will perform poorly.

         How it works?
         self.W = np.random.randn(D)/np.sqrt(D)
         randn() == draws samples from the standard normal and that's usually denoted as N(0,1)
         what happens when we multiply these samples by some number C. Well you can show thaat if you multiply random samples from the standard
         normal by a number c they will have mean zero and standard deviation c. In other words it's equivalent to sampling from the distribution
         and a zero c^2 N(0,C^2). Infact this is just the opposite of standardization and normalization.
         To standardize : Z = (X - mue)/ sigma (where mue is = mean and sigma = standard deviation).

         What does this code do?
         self.W = np.random.randn(D)/np.sqrt(D) means that we want weight w to have the mean 0 and variance 1/D.
         So why would we want to do that? If you don't do this then your loss per iteration may explode. well the reason it's exploding
         because those weights are just too large and if the weights are too large then that means their variance is too large.
         Which means we might be able to avoid this problem if we make the variance smaller. In some code people just multiply the weights
         by some constant like 0.01. Experimentation is the key there is no fixed rule to determine the initialization of the weights.
         It all depends on the data that you are working.
        '''
        self.W = np.random.randn(D)/np.sqrt(D)
        self.b = 0

        # create an array to hold that cost
        costs = []
        # keep track of best validation
        # Its gonna start as 1
        best_validation_error = 1
        for i in range(epochs):
            #calculate probability Y given x
            pY = self.forward(X)

            #gradient descent step
            self.W -= learning_rate*(X.T.dot(pY-Y) + reg*self.W)
            self.b -= learning_rate*((pY-Y).sum()+ reg*self.b)

            #in every 20 steps we are going to calculate the cost
            if i%20 == 0:
                pYvalid = self.forward(Xvalid)
                #calculating cost
                c = sigmoid_cost(Yvalid,pYvalid)
                costs.append(c)
                #calculating error rate
                e = error_rate(Yvalid,np.round(pYvalid))
                print (f"i: {i} , cost : {c} , error : {e}")
                #keep track of best validation error
                if e < best_validation_error:
                    best_validation_error = e
        print(f"best validation error : {best_validation_error}")

        if show_fig:
            plt.plot(costs)
            plt.show()
    def forward(self, X):
        return sigmoid(X.dot(self.W) + self.b)
    def predict(self, X):
        pY = self.forward(X)
        return np.round(pY)
    #This could be used for cross validation
    def score(self,X,Y):
        prediction = self.predict(X)
        return 1- error_rate(Y,prediction)

def main():
    #get binary data This does not automatically balances the classes of data for us
    X,Y = getBinaryData()
    # We are going to balance the classes of data here
    X0 = X[Y==0, :]
    X1 = X[Y==1, :]
    #repeate the occourances of data of X1
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0,X1])
    Y = np.array([0]*len(X0) + [1]*len(X1))

    #use our model the way we use scikitLearn
    model = LogisticModel()
    model.fit(X,Y,epochs=99999999,show_fig = True)
    model.score(X,Y)

if __name__ == '__main__' :
    main()
