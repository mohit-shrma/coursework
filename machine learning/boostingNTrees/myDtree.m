%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 10/25/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: learn decision tree on training data for passed depth and apply k-fold ...
       validation where k is passed as an argument to function
%}

function [] = myDtree(dataFileName, depth, numFolds)

load(dataFileName);

%as data is skewed randomly shuffle data
randInd = randperm(size(data, 1));
permData = data(randInd, :);
permLabels = labels(randInd, :);

%get size of dataset
sizeData = size(permData,1);

%size of data to be left for validation
leftDataSize = int16(sizeData/numFolds);

%store accuracy of learned parameters for each validation
trainErrorPcs = zeros(numFolds, 1);
testErrorPcs = zeros(numFolds, 1);

for iter=1:numFolds
    validStart = (iter-1) * leftDataSize + 1;
    validEnd = validStart + leftDataSize - 1;
    if validEnd > sizeData
        validEnd = sizeData;
    end
    
    validationData = permData(validStart:validEnd, :);
    validationLabels = permLabels(validStart:validEnd, :);
    
    if validEnd ~= sizeData 
        trainingData = [permData(1:validStart-1, :); permData(validEnd+1: ...
                                                   sizeData, :)];       
        trainingLabels = [permLabels(1:validStart-1); permLabels(validEnd+1: ...
                                                   sizeData, :)];
    else
        trainingData = permData(1:validStart-1, :);       
        trainingLabels = permLabels(1:validStart-1);       
    end
    
    %learn the decision tree and get the training and test error
    [trainErr, testErr] = dtree(trainingData, trainingLabels,...
                                validationData, validationLabels, ...
                                depth);
    
    testErrorPcs(iter) = testErr;
    trainErrorPcs(iter) = trainErr;
end

fprintf('\n*************************************************\n');

%trainErrorPcs
fprintf('\nTrain Errors: \n');
for iter=1:size(trainErrorPcs, 1)
    fprintf('%d\n', trainErrorPcs(iter));
end

fprintf('\nMean train error is %d\n', mean(trainErrorPcs));
fprintf('\nStandard deviation in train error is %d', std(trainErrorPcs));
fprintf('\n');

%testErrorPcs
fprintf('\nTest Errors: \n');
for iter=1:size(testErrorPcs, 1)
    fprintf('%d\n', testErrorPcs(iter));
end

fprintf('\nMean test error is %d', mean(testErrorPcs));
fprintf('\nStandard deviation in test error is %d', std(testErrorPcs));
fprintf('\n');


