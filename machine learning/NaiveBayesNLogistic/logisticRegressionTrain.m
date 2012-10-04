%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: apply logistic regression and compute weight parameters
by applying IRLS for weight
%}


function [weights, cost] = logisticRegressionTrain(data, labels)

%train data size
trainDataSize = size(data, 1);

%assuming label sent to this function will be transformed to 0-1
%change  class labels to 0-1
classes = unique(labels);

%add bias i.e a column vector of ones to data at start
data = [ones(size(data, 1), 1) data]; 

%number of features
numFeatures = size(data, 2);

%initialize weight parameters
weights = zeros(numFeatures, 1);

%choose iterations for which to run gradient descent
numIter = 100;

%learn parameters and cost by applying gradient descent on sigmoid 
cost = zeros(numIter, 1);

for iter=1:numIter
     
    Y = sigmoid(weights'*data')';
    R = diag(Y.*(1-Y));
    phi = data;
    H = phi'*R*phi;
    theInv = inv(H + 1e-6 *eye(size(H, 1)));
    z = phi*weights-(pinv(R)*(Y-labels));
    weights = theInv*phi'*R*z;
    
    cost(iter) = computeSigmoidCost(data, labels, weights);
    
    if (iter > 5) && ((cost(iter) - cost(iter-1)) <= 0.001)
        break;
    end
    
end
%plot(cost);



