%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: learn weights for each class
%}

function [weights] = oneVsAllTrain(data, labels)

%get classes
classes = unique(labels);

%number of features
numFeatures = size(data, 2);

%weight matrix containing column wise weight for each class
%note 1 added for bias
weights = zeros(numFeatures+1, size(classes, 1));

for classIter=1:size(classes, 1)
    currClass = classes(classIter);
    newLabels = (labels == currClass);
    [weightsVec, cost] = logisticRegressionTrain(data, newLabels);
    weights(:, classIter) = weightsVec;
end
