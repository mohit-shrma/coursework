%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: compute sigmoid cost
%}

function[cost] = computeSigmoidCost(data, labels, weights)

%data size
sizeData = size(data, 1);

h = sigmoid(weights'*data');

cost = (-1/sizeData)*((labels'*(log(h)')) + ((1-labels)'*(log(1-h)')));