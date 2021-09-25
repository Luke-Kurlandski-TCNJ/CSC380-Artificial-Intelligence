# 1.7
# Using Digit Dataset and K-nearest neighbor model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.datasets import load_digits
from sklearn import datasets
from sklearn.metrics import classification_report, accuracy_score



def showImage(i):
    digits = datasets.load_digits()
    plt.imshow(digits['images'][i],cmap='Greys_r')
    plt.show()
    

def one_point_seven():
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # KNN with K = 3 
    knnmodel = KNeighborsClassifier(n_neighbors=3)
    knnmodel.fit(X_train,y_train)

    # cross validation split data into k folds (5) 
    y_pred = cross_val_predict(knnmodel, X, y, cv=5)
    
    # testing accuracy for different K values
    score = []
    kValues = range(3, 20, 2)
    for k in kValues:
        knnmodel = KNeighborsClassifier(n_neighbors=k)
        y_pred = cross_val_predict(knnmodel, X, y, cv=5)
        accuracy = accuracy_score(y, y_pred)
        score.append(accuracy)
        print("k={:d}  accuracy={:0.5f} %".format(k, accuracy))

    plt.plot(kValues, score)
    plt.xlabel("K Values")
    plt.ylabel(" Accuracy ")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    knnmodel.fit(X_test,y_test)
    y_pred=knnmodel.predict(X_test)
    report= classification_report(y_test, y_pred)
    print(report)

    #testing
    test = np.array(digits['data'][1726])
    test1 = test.reshape(1,-1)
    showImage(1726)
    print(knnmodel.predict(test1))

  
if __name__ == "__main__":
    one_point_seven()
