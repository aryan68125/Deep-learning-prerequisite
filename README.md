# This repository holds the basic fundamentals of logistic regression
### We know that logistic regression is treated as a single neuron and a nural network is a collection of neurons. Hence we can say that logistic regression is the fundamental building block of deep neural networks
## Numpy axis --->
![](util_pictures_for_explaination_README/numpy_axis.jpg)

# What is Logistic regression : 
Logistic regression is a base line model. If you can get away with logistic regression then you should, It is very interpretable and stable . You don't have to do a lot of tuning to find the solution unlike neural networks. It is also fast and it is well established having been studied by statistitians for many decades.
![](util_pictures_for_explaination_README/util_images/logistic_regression/logistic_regression1.png)
Logistic regression can be seen as a model for the neuron. A combination of logistic regression is a neural network. Logistic regression is a linear model. by using feature engineering we can build powerful non-linear logistic regression model.
![](util_pictures_for_explaination_README/util_images/logistic_regression/logistic_regression2.png)
# Performance interpretation of a model : 
If our model only performs well on the training data not the test data we say that this model does not generalize well. So one way we can ensure that our model generalize well is to use regularization.
# linear classification :
![](util_pictures_for_explaination_README/util_images/linear_regression/linear_classification1.png)
As you can see we have a bunch of x in the left and a bunch of o in the right and we would like to separate them by drawing a line in a 2D plane.
equation of a line : <br>
y=mx+b <br>
Here y is the slope and b is the y intercept. <br>
0=ax+by+c <br>
you can see that if our x's and o's are split along a 45deg line that crosses the line y-intercept = 0 then ```a=1, b=-1, c=0``` should be the line. So in other words our line is 0 = x-y.
![](util_pictures_for_explaination_README/util_images/linear_regression/linear_classification2.png)
Let's say we have a test point h(2,1) --> x=2 and y=1 <br>
if we plug that into the equation h(2,1) =1>0 --> therefore weshould classify this new test point as an 'o'.
# How do we calculate the output of a neuron(Logistic Regression):
Just link in a brain of any organism is a collection of neurons which is a building block of a brain in a similar fashon,
a logistic regression is also a building black of a neural network.
#### Here is a diagram of a logistic regression (a single neuron) that is the building block of a neural network.
![](util_pictures_for_explaination_README/util_images/logistic_regression.jpg)
Here as you can see in the above diagram there are two circles with X's are multipliers which multiplies the x1 and x2 with the w1 and w2 here (w1 and w2 are the weights). and then there is another circle with nothing in that which is a summer and a non-linear transformer sigmoid(w1x1 + w2x2). So the unique thing about the logistic regression is the circle that comes in front of the output y. It applies logistic function or the sigmoid function.
#### Here is a diagram of a Sigmoid or Logistic function:
![](util_pictures_for_explaination_README/util_images/sigmoid_function.png)
![](util_pictures_for_explaination_README/util_images/sigmoid_function_formula.png)
Sigmoid Function has a finite limit as X approaches infinity and a finite limit as X approaches minus infinity.Sigmoid function goes from 0 to 1 and it's Y-intercept is 0.5.There are two commonly used Sigmoid functions that are used in AI/ML,
one is hyperbolic tangent or tanh(x) which goes from (-1,1) and it's Y=intercept is 0 and another one is a sigmoid function denoted by a letter called sigma as we have seen above.
#### Here is a diagram of tanh function or hyperbolic tangent function:
![](util_pictures_for_explaination_README/util_images/tanh_function.png)
![](util_pictures_for_explaination_README/util_images/tanh_function_formula.png)
So we can combine these to say the output of a logistic regression is --> sigma of the inner products of the weights times X σ( w^t.x). So if the inner product of w and x is very positive then we will get the number that is very close to 1. If the inner product of W and X is very negative we will get the number that is very close to 0. If the output of sigmoid function is 0.5 then the value of inner product is 0 which means we are right at the boundary between the two classes (The probability of belognging to either classes is 50%).

## Difference between Logistic regression and general linear classifier:
We have this logistic function (Sigmoid function) at the end which gives us the number between 0 and 1. here we can say that anything which gives us a number above 0.5 gives us class 1 and anything below 0.5 gives us class 0. the value of sigmoid for input 0 is 0.5.

# What does the output of Logistic regression actually means?
![](util_pictures_for_explaination_README/util_images/formula_logisticregression.jpg).  
##### The output of the logistic regression is a sigmoid. form the sigmoid we are going to get a number between 0 and 1. In deep learning it has a nice and intuitive interpretation.
##### First let's recall what we are trying to do during classification.
![](util_pictures_for_explaination_README/util_images/what_is_classification.jpg)
##### We have some red dots and some blue dots and we have a line that separates them. Each dot is represented by a feature vector X and it's color or it's label is represented by a label Y. as per conventions Y has a value 0 or 1. here if Y=0 --> red if Y=1 --> blue.  
![](util_pictures_for_explaination_README/util_images/output_logisticregression.jpg)
##### The output of logistic regression is a number between 0 and 1, we interpret these as a probability that y=1 given x.
![](util_pictures_for_explaination_README/util_images/output_interpretation.jpg)
##### And so this gives us a handy way of making predictions. If p(y=1 | x) > p(y=0 | x): predict class 1 ,else: predict class 0.  


## Problems related to Logistic Regression :
### Donut problem :
![](util_pictures_for_explaination_README/donut_problem.png)
#### So Linear regression might not be good for this donut problem because there is no line that can separate the two dataset here in this case I's the Yellow and the Purple data point sets
```
TRICK OF THE DONUT PROBLEM:
       The trick with the donut problem is -> We are going to create yet another column which represents the radius of a point
       this will make your data points linearly separable

Code->
       r = np.zeros((N,1)) # creating a N size and 1 dimension of numpy array
       #manually calculate the radiuses
       for i in range(N):
           r[i] = np.sqrt(X[i,:].dot(X[i,:]))
       #Now then I do my concatenation the ones and the radiuses and It's all togeather
       #axis=1 -> adding ones npArray, r npArray and X npArray by column
       Xb = np.concatenate((ones,r,X), axis = 1)
```
#### In general you will have to experiment a little bit to find the right number for these values or you could use something like cross-validation
```
#set the learning rate to 0.0001
learning_rate = 0.0001
```

#### Calculating weights
```
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
    #re-calculating the output
    Y = sigmoid(Xb.dot(w))
```

#### Print the final weights and Final classification of the model
```
print(f"Final weights : {w}")
#when we are classifying we are actually rounding the output from the sigmoid function here in this case it's Y
print(f"Final classification rate : {1-np.abs(T-np.round(Y)).sum() / N}")
```

#### Explaining the output of the model
```
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
```

### XOR problem :
![](util_pictures_for_explaination_README/Logistic_regression_XOr_problem.png)
#### Here you can see that there are 4 data points 2 are from class yellow and 2 are from class purple. So the trouble using logistic regression here is that you can't really find the line that will give you a perfect classification
#### REASON WHY LOGISTIC REGRESSION ISN'T APPLICABLE IN THIS XOR CASE ->
Here you can see that there are 4 data points 2 are from class yellow and 2 are from class purple.
So the trouble using logistic regression here is that you can't really find the line that will give you a perfect classification
#### TRICK TO SOLVE XOR PROBLEM VIA LOGISTIC REGRESSION ->
So the Trick is with the XOr problem is we're again going to add another dimension to our input
So We are going to turn it into a 3D peoblem instead of a 2D problem and then we can draw a plane between the two datasets
--> If we multiply the x and y to a new variable we can make the data linearly separable
```
xy = np.matrix(X[:,0]*X[:,1]).T
#Now then I do my concatenation the ones and the xy npArray and It's all togeather
#axis=1 -> adding ones npArray, xy npArray and X npArray by column
Xb = np.array(np.concatenate((ones,xy,X), axis=1))

'''
The rest of the code will be the same
'''
```
### These past two example bring out a really interesting point ->
#### You saw that we could apply the logistic regression to some compex problems by manual feature engineering. So we looked at the data and we determine some features that we could calculate the inputs that would help us improve our classification. Now in machine learning ideally the machine would be able to learn these things and so that is precisely what neural networks do. So in the future we will apply these problems and implement a neural networks that will automatically learn the features like this

## Gradient Descent :
#### Final output after applying gradient descent to find weights for the logistic regression function instead of randomly selecting weights
![](util_pictures_for_explaination_README/gradient_descent_1.png)
#### This is the graph that helps you visualize how a gradient descent is done
![](util_pictures_for_explaination_README/gradient_descent_2.jpg)
### Note->
Logistic function = Sigmoid funtion
#### The Sigmoid funtion graph is shown as below ->
![](util_pictures_for_explaination_README/sigmoid_funtion_graph.png)

# Some points to remember when analyzing data in machine learning:
## Data pre-processing:
### Handle time column in a database or an excel sheet
- When dealing with time which is cyclical in nature. We catagorize the time into different categories.
- The categories may look something like this 
    - 0 for 12 am to 6 am 
    - 1 for 6 am to 12 pm
    - 2 for 12 pm to 6 pm
    - 3 for 6 pm to 12 am
- The reason we divided time into buckets is because we assume that the users in the same bucket will behave similary. So this helps out when these types of columns end up in the training data when training our machine learning model.
### One hot encoding:
![](util_pictures_for_explaination_README/data_pre_processing/one_hot_encoding.png)
- You can't feed your category type data into your logistic regression model or your neural network model because these work on numerical vectors.
- For information visit this link:
    - [Explain numerical vectors in this context and why we can't feed categorical data in logistic regression or neural network](extra_docs_for_readme/numerical_vectors_one_hot_encoding.md)
- In order to solve this problem we use One Hot Encoding:
    - This simply means that if we have 3 different categories then we will use three different columns to represent them. We then set each column that represents the category for each sample to one.
### Handle binary categories
- Technically we could turn them into two different columns using one hot encoding but we don't necessarily need to.
    - [Explain numerical vectors in this context and why we can't feed categorical data in logistic regression or neural network](extra_docs_for_readme/handle_binary_categories.md)
    - [What does it mean to absorb the off effect into the bias term in the context of binary categories](extra_docs_for_readme/absorb_the_off_effect_into_the_bias_term.md)
    - [what is multicollinearity](extra_docs_for_readme/multicollinearity.md)

# Sentiment Analysis with Logistic Regression
## Case 1 : Without L1 Regularization
#### Findings during my testing ->
1. SciKitLearn Logistic regression class performance was around 0.72
   ```
   Using Logisitc Regression from scikitLearn
   Classification rate : 0.72
   ```
2. Logistic regression when applied from scratch with gradient descent and not using LogisticRegression Class from ScikitLearn I got performance of
   0.76
```
   Final train classification rate : 0.8694736842105263
   Final test classification rate : 0.76
   ```
3. This is the train vs test cost graph for 20,000 epox
![](util_pictures_for_explaination_README/sentiment_analysis_project/sentiment_analysis_without_L1_regularization/sentiment_analysis_logistic_regression_without_L1_regularization_20000_epox_train_vs_test_cost_graph.png)
4. This is the weight graph for 20,000 epox
![](util_pictures_for_explaination_README/sentiment_analysis_project/sentiment_analysis_without_L1_regularization/sentiment_analysis_logistic_regression_without_L1_regularization_20000_epox_weight_graph.png)
6. Learning rate is set to 0.001
## Case 2 : With L1 Regularization
#### Findings during my testing ->
1. SciKitLearn Logistic regression class performance was around 0.61
   ```
   Using Logisitc Regression from scikitLearn
   Classification rate : 0.61
   ```
2. Logistic regression when applied from scratch with gradient descent with L1 Regularization and not using LogisticRegression Class from ScikitLearn I got performance of
   0.56
   ```
   Final train classification rate : 0.6652631578947369
   Final test classification rate : 0.56
   ```
3. This is the train vs test cost graph for 20,000 epox
![](util_pictures_for_explaination_README/sentiment_analysis_project/sentiment_anamysis_with_L1_regularization/sentiment_analysis_logistic_regression_with_L1_regularization_20000_epox_train_vs_test_cost_graph.png)
4. This is the weight graph for 20,000 epox
![](util_pictures_for_explaination_README/sentiment_analysis_project/sentiment_anamysis_with_L1_regularization/sentiment_analysis_logistic_regression_with_L1_regularization_20000_epox_weight_graph.png)
6. Learning rate is set to 0.001 and L1 penalty was set to 1.0

## NOTE :
### Here in L1 regularization we expect sparse solutions, i.e. some weights will be pushed to exactly zero
## Case 3 : Logistic regression with hyperparameter tuning
1. SciKitLearn Logistic regression class with hyper parameter tuning performance was around 0.840
The best hyper parameters found by GridSearchCV (C=4.281332398719396, penalty='none', solver='sag')
   ```
   Findings when looking through the console
   best model hyperparameters for logisticRegressionModel :
   LogisticRegression(C=0.23357214690901212, penalty='none', solver='sag') Accuracy : 0.890
   ```
   ```
   best hyperparameters for logistic regression model : LogisticRegression(C=0.23357214690901212, penalty='none', solver='sag')
   Model Predictions : [1. 0. 0. 1. 0. 0. 1. 1. 1. 1. 1. 0. 0. 1. 1. 1. 1. 0. 0. 0. 0. 1. 1. 0.
   0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 1. 1. 1. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 1.
   1. 1. 1. 0. 1. 0. 1. 1. 0. 1. 0. 0. 1. 1. 1. 1. 0. 0. 0. 1. 1. 0. 1. 0.
   0. 1. 1. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 0. 1. 1. 0. 0. 1. 0. 1. 1. 0. 0.
   1. 1. 1. 0.]
   Targets or Ytest : [1. 0. 0. 1. 0. 1. 1. 0. 1. 1. 1. 0. 0. 1. 1. 1. 1. 1. 0. 0. 0. 0. 1. 0.
   0. 0. 0. 1. 0. 0. 0. 1. 0. 1. 1. 1. 1. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 1.
   1. 1. 1. 0. 0. 0. 1. 1. 0. 0. 0. 0. 1. 1. 1. 1. 0. 0. 1. 1. 1. 0. 1. 0.
   0. 1. 1. 1. 0. 1. 0. 1. 1. 1. 0. 1. 0. 0. 1. 1. 0. 1. 1. 0. 1. 1. 0. 0.
   0. 1. 1. 0.]
   Accuracy : 0.890
   ```
## Note :
### The code related to case 3 is incorporated in sentiment_analysis_without_L1_regularization.py file. I did not make a separate file case 3 demonstration SORRY.
## Conclusion :
#### As you can see the improvements are quite clear performance of case 1 < case 2 < case 3. I am sure there is still some room for improvements. My research is under progress. I will update this once I find anything on this topic. For now I will stop it here.
# Facial expression recognition
## Class imbalance :
#### Here in facial expression recognition problem we have 547 samples from class 1 and 4953 samples from class 0. This is a severe problem because this means that your model will suffer severe hits during its learning process and it will try to classify the unseen data as class 0 most of the time.
## Solution to class imbalance :
####  Suppose we have 1000 samples from class 1 and 100 samples from class 2
1. Pickup 100 samples from class 1,now we have 100 from class 1 vs 100 from class 2.
2. Repeat class 2 10 times , now we have 1000 from class 1 vs 1000 from class 2.
#### NOTE that these 2 strategies both have the same expected error rate. Method 2 is much better because we know variance also play an important role when training a model on a dataset. Hence promoting us to feed more data to the model for better output results during training.
## Other accuracy measures :
#### Note that there are other ways to measure the acurracy as well that allow for class imbalances. These are used in the medical field and in information retrieval and try to take into account both classes. This assumes we are doing binary classification.
#### The basic idea behind this is :->
1. Maximize both true positives and true negatives
2. Minimize both false positives and false negatives
![](util_pictures_for_explaination_README/true_positive_ture_negative.png)
#### What is false positive and false negative ?
1. False positive : A false positive is when you predict the positive class but the actual label is the negative class.
2. False negative : A false negative is when you predict the negative class but the actual label is the positive class.
#### Note : The accuracy is (TP + TN)/(TP+TN+FP+FN) where TP is true positive , TN is true negative , FP is false positive and FN is false negative.
#### In the medical field we use sensitivity and specificity
1. Sensitivity or True positive rate : TP/(TP+FN)
2. specificity or True negative rate : TN/(TN+FP)
#### In information retireval we use precision and recall
1. Precision : TP/(TP+FP)
2. Recall or sensitivity : TP/(TP+FN)
#### Total score is F1-Score Combines precision and recall into a balanced measure F1 = 2*(precision*recall)/(precision+recall). This is the harmonic mean of precision and recall
## Final score of Facial expression recognition:
#### The validation Score after training of logistic regression model is complete is -> ```best validation error : 0.14```
#### The graph after the training of logistic regression model is complete ->
![](util_pictures_for_explaination_README/facial_recognition/emotion_recognition_logistic_regression.png)







