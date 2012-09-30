%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: apply linear discriminant to project in lesser dimension, ...
       then uses generative modelling to learn parameters and return ...
       them
%}

function [projectedMeans, sharedCovariance, classPriors, weightVec] = fisherTrain(data, labels)

%get the class labels in array
classes = unique(labels);

%get size of dataset
sizeData = size(data,1);

%get the num of features
numFeatures = size(data,2);

%init mean data vector
meanData = mean(data);

%compute within class covariances
withinClassCovariance = zeros(numFeatures, numFeatures);

%compute individual class covariances, means and prior    
[classCovariances, classMeans, classSize, classPriors] = classCovarianceNMeans(data, labels, classes);

%between class covariance
bwClassCovar = zeros(numFeatures, numFeatures);
for iter=1:size(classes, 1)
    %compute within class covariances, by summing classCovariances
    withinClassCovariance = withinClassCovariance + ...
        (reshape(classCovariances(iter,:), numFeatures, ...
                 numFeatures)*classSize(iter));
    
    %add this class variance from data to between class variance
    classVarianceVec = (classMeans(iter, :) - meanData)';
    classVariance = classSize(iter)*(classVarianceVec*classVarianceVec');
    bwClassCovar = bwClassCovar + classVariance;
end

%to maximize: J(w)=Tr{inv(W * withinClassCovar * W') (W * bwClassCovar * W')}
%weight values determined by eigen vector of
%(inv(withinClassCovar)*bwClassCovar) 
%correspond to D' largest eigen values where D' is dimension of
%projection, Max(D') = No. of classes - 1
%in iris case Max(D') = 3 - 1

%get top numClasses-1 as weight vectors
numWVec = size(classes, 1) - 1;



[weightVec, eigenVal] = eigs(inv(withinClassCovariance)*bwClassCovar, numWVec);

%project data into into new space using above weight vectors
projectedData = zeros(sizeData, numWVec);
for iter=1:sizeData
    projectedData(iter, :) = (weightVec' * data(iter, :)')';
end

%DEBUG:plot projected data to see whether separated correctly
%scatter(projectedData(:,1), projectedData(:,2), 5, labels)

%{
  now we will use maximum-likelihood solution to estimate parameters ...
      of gaussian fitting class conditionals
  mean is respective class's projected data mean
  priors are fraction of points in respective class (class priors ...
                                                    are computed above)
  shared covariance is weighted average of covariance matrices ...
      associated with classes
%}

numProjectedFeatures = size(projectedData, 2);

[projectedCovariances, projectedMeans, classSize, classPriors] = ...
    classCovarianceNMeans(projectedData, labels, classes)

%compute shared covariance
sharedCovariance = zeros(numProjectedFeatures, ...
                         numProjectedFeatures);
for iter=1:size(classes,1)
    sharedCovariance = sharedCovariance + classPriors(iter)* ...
        (reshape(projectedCovariances(iter, :), numProjectedFeatures, ...
                 numProjectedFeatures));
end

