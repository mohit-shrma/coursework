"""
   CSci5512 Spring'12 Homework 4
   login: sharm163@umn.edu
   date: 4/30/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: perceptron
"""

import sys
from numpy import *
#import matplotlib.pyplot as plt
#from pylab import *

#mean normalize
#X = (X -mu) / range, where range = max -min
def meanNormalizeColumnWise(X):
    numCols = X.shape[1]
    for i in range(numCols):
        meanXi = mean(X[:, i])
        minXi = min(X[:, i])
        maxXi = max(X[:, i])
        rangeXi = maxXi - minXi
        X[:, i] = (X[:, i] - meanXi)/rangeXi
    return X

def gradientDescentSigmoid(X, Y, theta, alpha, numIter):
    cost = zeros((numIter, 1))
    trainSize = X.shape[0]
    for i in range(numIter):
        K = alpha/trainSize
        #K = alpha/((i+1)*trainSize)
        delta = sigmoid(dot(X, theta)) - Y
        for j in range(len(theta)):
            theta[j] -= K * dot(delta.T, X[:, j])
        cost[i] = computeSigmoidCost(X, Y, theta)
    return (theta, cost)

def sigmoid(X):
    Z = float(1)/(1+exp(-1*X))
    return Z

#cost = -1/m * [ sum<m> { y log(h(xi)) + (1-y) log(1 - h(xi))}  ]
def computeSigmoidCost(X, Y, theta):
    numEx = shape(X)[0]
    h = sigmoid(dot(X, theta))
    cost = (-1*float(1)/numEx)*( dot(Y.T, log(h)) \
                                 + dot((1-Y).T, log(1-h)) )
    return cost

def predictSigmoid(theta, test):
    h = sigmoid(dot(test, theta))
    if h > 0.5:
        return 1
    else:
        return 0
    
def getSigmoidAccuracy(X, Y, theta):
    predictions = zeros((X.shape[0],1))
    numTest = X.shape[0]
    equalCount = 0
    for i in range(numTest):
        predictions[i] = predictSigmoid(theta, X[i,:])
    res = (predictions == Y)
    for k in res:
        if k:
            equalCount += 1
    return 100*float(equalCount)/numTest


def main():
    #create an array of examples to use
    #column represents the atributes
    #['alt' {0,1}, 'bar' {0,1}, 'fri' {0,1}, 'hun' {0,1}, 'pat' {none:0, some:1,
    #full:2}, 'price' {$:0, $$:1, $$$:2}, 'rain' {0,1},
    #'res' {0,1}, 'type' {french:0, thai:1, burger:2, italian:3},
    #'est' {'0-10': 0, '30-60':1, '10-30':2, '>60':3}]
    X = array([[1, 0, 0, 1, 1, 2, 0, 1, 0, 0]], dtype=float)
    X = vstack((X,
              [[1, 0, 0, 1, 2, 0, 0, 0, 1, 1]]))
    X = vstack((X,
              [[0, 1, 0, 0, 1, 0, 0, 0, 2, 0]]))
    X = vstack((X,
              [[1, 0, 1, 1, 2, 0, 1, 0, 1, 2]]))
    X = vstack((X,
              [[1, 0, 1, 0, 2, 2, 0, 1, 0, 3]]))
    X = vstack((X,
              [[0, 1, 0, 1, 1, 1, 1, 1, 3, 0]]))
    X = vstack((X,
              [[0, 1, 0, 0, 0, 0, 1, 0, 2, 0]]))
    X = vstack((X,
              [[0, 0, 0, 1, 1, 1, 1, 1, 1, 0]]))
    X = vstack((X,
              [[0, 1, 1, 0, 2, 0, 1, 0, 2, 3]]))
    X = vstack((X,
              [[1, 1, 1, 1, 2, 2, 0, 1, 3, 2]]))
    X = vstack((X,
              [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]))
    X = vstack((X,
              [[1, 1, 1, 1, 2, 0, 0, 0, 2, 1]]))

    #normailize column wise i.e feature wise across above examples
    X = meanNormalizeColumnWise(X)

    #add a column vector of 1's at start as X0
    X = hstack(( ones((X.shape[0] ,1)), X ))
    
    #training set result vector
    Y = array([[1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]).T

    #initialize fitting parameters
    theta = zeros((X.shape[1],1), dtype=float)

    #choose learning rate
    alpha = .1

    #choose iterations for which to run gradient descent
    numIter = 5000

    #apply sigmoid gradient descent 
    (theta, cost) = gradientDescentSigmoid(X, Y, theta, alpha, numIter)

    #plot cost to see if gradient descent working fine
    #plot(cost)
    #show()
    #print X
    #print theta.shape

    #print X[10,:]
    #print predictSigmoid(theta, X[11,:])
    #print Y[11]

    print "***** prediction accuracy *****"
    accuracy = getSigmoidAccuracy(X, Y, theta)
    print 'training set error: '+ str(100-accuracy) + '%'
    
if __name__ == '__main__':
    main()
