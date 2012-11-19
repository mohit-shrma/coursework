%{
   CSci5525 Fall'12 Homework 3
   login: sharm163@umn.edu
   date: 11/17/2012
   name: Mohit Sharma
   id: 4465482
   algorithm:predict the label for data using bagged classifiers
%}
function [predictedLabel] = baggedPrediction(baggedClassifiers, ...
                                             dataVec)

%num of bagged classifiers
numBaggedClassifiers = size(baggedClassifiers, 1);

%vector to store predicted labels
predictedLabels = zeros(numBaggedClassifiers, 1);

%predict labels from each bagged classifier
for iter=1:numBaggedClassifiers
    predictedLabels(iter) = baggedClassifiers(iter).predictLabel(dataVec);
end

%take the majority value of labels as predicted labels
predictedLabel = mode(predictedLabels);

