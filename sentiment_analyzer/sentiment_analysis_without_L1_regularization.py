import nltk
nltk.download('punkt')
import numpy as np
from sklearn.utils import shuffle

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
'''
Note that I have turned off the warnings as the GridSearchCV() function tends to generate quite a bit of warnings.
'''
import warnings
warnings.filterwarnings('ignore')

wordnet_lemmatizer = WordNetLemmatizer()

# from http://www.lextek.com/manuals/onix/stopwords1.html
'''
Set up a variable that will have all the possible stop words
These stop words are required to compare the sentences and remove them before we feed the sentences to the tokenizer
Why are we doing this?
Because stop words do not add any value when predicting a positive or a negative sentiment from sentences (I't reviews in this case)
'''
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# note: an alternative source of stopwords
# from nltk.corpus import stopwords
# stopwords.words('english')

# load the reviews which is in the xml format for this we will have to use Beautiful soup sml parser
# here we are using html5lib parser from beautiful soup xml parser
positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), features="html5lib")
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read(), features="html5lib")
negative_reviews = negative_reviews.findAll('review_text')



# first let's just try to tokenize the text using nltk's tokenizer
# This will convert
# let's take the first review for example:
# t = positive_reviews[0]
# nltk.tokenize.word_tokenize(t.text)
# notice how it doesn't downcase, so It != it
# not only that, but do we really want to include the word "it" anyway?
# you can imagine it wouldn't be any more common in a positive review than a negative review
# so it might only add noise to our model.
# so let's create a function that does all this pre-processing for us
'''
Solving an NLP problem is a multi-stage process. We need to clean the unstructured text
data first before we can even think about getting to the modeling stage. Cleaning the data consists of a few key steps:
    Word tokenization
    Predicting parts of speech for each token
    Text lemmatization
    Identifying and removing stop words, and much more.
Tokenization is one of the most common tasks when it comes to working with text data.

But what does the term ‘tokenization’ actually mean?
    Tokenization is essentially splitting a phrase, sentence, paragraph, or an entire text document into smaller units, such as individual
    words or terms. Each of these smaller units are called tokens.The tokens could be words, numbers or punctuation marks. In tokenization,
    smaller units are created by locating word boundaries

what are word boundaries?
These are the ending point of a word and the beginning of the next word. These tokens are considered as a first step for stemming and
lemmatization
Before processing a natural language, we need to identify the words that constitute a string of characters. That’s why tokenization is the most
basic step to proceed with NLP (text data). This is important because the meaning of the text could easily be interpreted by analyzing the
words present in the text.
Let’s take an example. Consider the below string:
“This is a cat.”
What do you think will happen after we perform tokenization on this string? We get [‘This’, ‘is’, ‘a’, cat’].

We can use this tokenized form to:
    Count the number of words in the text
    Count the frequency of the word, that is, the number of times a particular word is present

NLTK contains a module called tokenize() which further classifies into two sub-categories:
    Word tokenize: We use the word_tokenize() method to split a sentence into tokens or words
    Sentence tokenize: We use the sent_tokenize() method to split a document or paragraph into sentences

code ->
from nltk.tokenize import word_tokenize
    text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet
    species by building a self-sustaining city on Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed
    liquid-fuel launch vehicle to orbit the Earth."""
    word_tokenize(text)
Output: ['Founded', 'in', '2002', ',', 'SpaceX', '’', 's', 'mission', 'is', 'to', 'enable',
         'humans', 'to', 'become', 'a', 'spacefaring', 'civilization', 'and', 'a',
         'multi-planet', 'species', 'by', 'building', 'a', 'self-sustaining', 'city', 'on',
         'Mars', '.', 'In', '2008', ',', 'SpaceX', '’', 's', 'Falcon', '1', 'became',
         'the', 'first', 'privately', 'developed', 'liquid-fuel', 'launch', 'vehicle',
         'to', 'orbit', 'the', 'Earth', '.']
Notice how NLTK is considering punctuation as a token? Hence for future tasks, we need to remove the punctuations from the initial list.
'''
def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens


# create a word-to-index map so that we can create our word-frequency vectors later
# let's also save the tokenized versions so we don't have to tokenize again later
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []
orig_reviews = []

#saving original reviews and it's token (Positive reviews)
for review in positive_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
#saving original reviews and it's token (Negative reviews)
for review in negative_reviews:
    orig_reviews.append(review.text)
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

print("len(word_index_map):", len(word_index_map))

# now let's create our input matrices
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1) # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum() # normalize it before setting label
    x[-1] = label #setting up the label
    return x

'''setting up the number of samples for our model here'''
N = len(positive_tokenized) + len(negative_tokenized)
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later
# using np.zeros create anea numpy array of shape N by D+1 where N is the number of samples and D+1 is the number of features
# here D is len(word_index_map)
'''number of features'''
D = len(word_index_map)
data = np.zeros((N, D + 1))

#adding positive_tokenized and negative_tokenized to the data numpy array
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1

# shuffle the data and create train/test splits
# try it multiple times!
orig_reviews, data = shuffle(orig_reviews, data)

'''
here we are saving data numpy array to X by selecting all rows and selecting all columns excluding the last column that's what "[:,:-1]" is doing
It's selecting all the columns from 0 to second last position excluding the last position of the column in the data array
'''
X = data[:,:-1]
'''
here we are saving data numpy array to Y by selecting all the rows and onlLogisticRegressiony selecting the very last column of the data numpy array that's what
"[:,-1]" is doing it's only selecting the very last column and all the rows from data and saves it to Y variable
Y = Targets
'''
Y = data[:,-1]

# last 100 rows will be test
# creating a train batch by excluding the last 100 rows
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
#creating a test batch by only selecting the last 100 rows
Xtest = X[-100:,]
Ytest = Y[-100:,]

#using a pre build LogisticRegression model from scikit learn
#defining our model here
print("Using Logisitc Regression from scikitLearn without hyper parameters tuning")
model = LogisticRegression()
model.fit(Xtrain , Ytrain)
print(f"Classification rate : {model.score(Xtest,Ytest)}")
'''
So we can look at the weights that each word has, to see if that word has positive or negative sentiment.
So we don't wanna look at all the weights. We just wanna look at the weights that are very far away from zero.
'''
#so I am going to set a thresh hold
threshold = 0.5
#I am gonna loop through every word in our map
for word, index in word_index_map.items():
    #I am gonna get the weight from the model
    weight = model.coef_[0][index]
    # if weight is greater than threshold or weight is less than -ve of the threshold print the word and the weight
    if weight > threshold or weight < -threshold:
        print(f"word,weight : {word,weight}")

print("Using Logisitc Regression from scikitLearn with hyper parameters tuning")
logisticRegressionModel = LogisticRegression()
param_grid = [
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000, 2500, 5000]
    }
]
# GridSearchCV(multi_class='multinomial') Pass multi_class as multinomial if you want to use logistic regression to classify multiple classes
clf = GridSearchCV(logisticRegressionModel, param_grid = param_grid, cv = 3, verbose=4, n_jobs=-1)
#selecting the best logistic regression classifier
best_clf = clf.fit(Xtrain , Ytrain)
predict = best_clf.predict(Xtest)
print(f"best hyperparameters for logistic regression model : {best_clf.best_estimator_}")
print(f"Model Predictions : {predict} \n Targets or Ytest : {Ytest}")
print (f'Accuracy : {best_clf.score(Xtest,Ytest):.3f}')
'''
Findings when looking through the console
best model hyperparameters for logisticRegressionModel :
LogisticRegression(C=4.281332398719396, penalty='none', solver='sag') Accuracy : 0.840
best hyperparameters for logistic regression model : LogisticRegression(C=0.23357214690901212, penalty='none', solver='sag') Accuracy : 0.890
'''

print("Using my own Logistic Regression model using numpy without L1 regularization")
#we can use that "dimensionality of the dataset" to initialize the weights of the Logistic regression model
W = np.random.randn(D) #weights setting up to be at random
# b is the baised term so that's the scalar
b=0 #biase

#sigmoid function which generates the output
def sigmoid(a):
    return 1/(1+np.exp(-a))

#forward function
#this function is responsible for making predictions using sigmoid function
#passing on the X : Data Matrix, W : Weight matrix , b : bias
def forward(X,W,b):
    return sigmoid(X.dot(W) + b)

#function that will give us the classification rate
# it returns the mean of Y==P
def classificationRate(Y,P):
    #This may look like it will return an array of boolean but it actually returns 0s and 1s
    #It will divide the number of correct prediction by total number of predictions
    return np.mean(Y==P)

#corss_entropy function this is gonna take in Targets which is Ytrain and Ytest and PofY given X -> pYtrain or pYtest
#error function or objective funtion
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

# plot the true w vs the w we found so that we can comare them
plt.plot(W,label = 'w map')
plt.legend()
plt.show()
