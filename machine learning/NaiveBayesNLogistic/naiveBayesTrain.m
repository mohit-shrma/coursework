%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: apply naive bayes to passed data and return learned parameters
%}

function [] = naiveBayesTrain(data, labels)

%get the class labels in array
classes = unique(labels);

%get size of dataset
sizeData = size(data,1);

%get the num of features
numFeatures = size(data,2);

%compute individual class covariances, means and prior    
[classCovariances, classMeans, classSize, classPriors] = classCovarianceNMeans(data, labels, classes);


