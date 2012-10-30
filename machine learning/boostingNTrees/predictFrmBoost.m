%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 10/25/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: use boosting weights and passed stumps to predict label
%}

function [label] = predictFrmBoost(dataVec, learnedStumps, ...
                                   boostClassifierWeights)

%initializes weighted classifiers sum
weightedClassifiersSum = 0;

%sum up the weighted boosted classifier label
for iter=1:size(learnedStumps, 1)
    weightedClassifiersSum += boostClassifierWeights(iter)*...
        (predictFrmDstump(trainingData(validIter, :),...
                          learnedDStumps(iter));
end

%return labels based on sign
if weightedClassifiersSum > 0:
    return 1;
else
    return -1;
end

