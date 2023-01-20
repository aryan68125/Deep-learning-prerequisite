import numpy as np
import matplotlib.pyplot as plt
from util import getData

label_map = ['Anger', 'Disgust' , 'Fear' , 'Happy' , 'Sad' , 'Surprise' , 'Neutral']
def main():
    '''
    It loads the data and preprocesses it
    '''
    X,Y = getData(balance_ones = False)
    while True :
        for i in range(7):
            x,y = X[Y==i], Y[Y==i]
            N = len(y)
            j = np.random.choice(N)
            '''
             Color map to grey since it's a grey scale image.
            '''
            plt.imshow(x[j].reshape(48,48),cmap = 'gray')
            plt.title(label_map[y[j]])
            plt.show()
        '''
        Because it's an infinite loop We are going to write the code to promp the user to quite the loop.
        '''
        prompt = input('Quite? Enter Y:\n')
        if prompt == 'Y':
            break
if __name__ == '__main__':
    main()
