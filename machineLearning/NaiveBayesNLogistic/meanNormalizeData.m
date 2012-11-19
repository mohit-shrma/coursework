%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: normalize data features
%}

function [data] = meanNormalizeData(data)
numFeatures = size(data, 2);
for iter=1:numFeatures
    meanFeature = mean(data(:, iter));
    minFeature = min(data(:, iter));
    maxFeature = max(data(:, iter));
    range = maxFeature - minFeature;
    data(:, iter) = (data(:, iter) - meanFeature)/range; 
end
