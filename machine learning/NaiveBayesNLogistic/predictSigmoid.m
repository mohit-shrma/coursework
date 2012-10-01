%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: apply given weights to i/p to predict label (0,1)
%}

function [predLabel] = predictSigmoid(weights, dataRowVec)

%append row vector with 1 on top
dataRowVec = [1 dataRowVec];

%compute sigmoid
h = sigmoid(weights'*dataRowVec');

%predict label
if h > 0.5
    predLabel =  1;
else
    predLabel = 0;
end

