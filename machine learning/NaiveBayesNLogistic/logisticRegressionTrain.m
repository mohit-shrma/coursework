%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: apply logistic regression and compute weight parameters
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
weights = ones(numFeatures, 1);

%choose iterations for which to run gradient descent
numIter = 800;

%learn parameters and cost by applying gradient descent on sigmoid 
cost = zeros(numIter, 1);

for iter=1:numIter
     
    Y = sigmoid(weights'*data')';
    R = diag(Y.*(1-Y));
    phi = data;
    theInv = pinv(phi'*R*phi);
    z = phi*weights-(pinv(R)*(Y-labels));
    weights = theInv*phi'*R*z;
     
    %K = alpha/trainDataSize;
    %K= alpha;
    %delta = sigmoid((weights'*data'))' - labels;
    
    %update wieights
    %for featureIter=1:numFeatures
    %    weights(featureIter) = weights(featureIter) - (K * (delta'*data(:, featureIter)));
    %end
    
    cost(iter) = computeSigmoidCost(data, labels, weights);
    
    if (iter > 1) && ((cost(iter) - cost(iter-1)) <= 0.0001)
        break;
    end
    
end
%plot(cost);



