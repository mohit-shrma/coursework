%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: apply naive bayes to passed data and return learned parameters
%}

function [gaussianParams, classPriors] = naiveBayesTrain(data, labels)

%get number of features
numFeatures = size(data, 2);

%get the class labels in array
classes = unique(labels);


%store gaussian parameter for each features
gaussianParams = zeros(size(classes,1), numFeatures, 2);

%class priors
classPriors = zeros(size(classes,1), 1);

for iter=1:size(classes)
    currClassLabel = classes(iter);
    ind = find(labels == currClassLabel);
    currClassData = data(ind, :);
    classPriors(iter) = size(currClassData, 1)/size(data, 1);
    for featureIter=1:numFeatures
        %TODO: how to handle sigma = 0
        [mu, sigma] = normfit(currClassData(:, featureIter));
        gaussianParams(iter, featureIter, 1) = mu;
        gaussianParams(iter, featureIter, 2) = sigma;
    end
end

