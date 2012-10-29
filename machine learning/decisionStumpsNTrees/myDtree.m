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
errorPcs = zeros(numFolds, 1);

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
    
    errorPcs(iter) = testErr;

end

fprintf('\n*************************************************\n');
fprintf(dataFileName);
%errorPcs
fprintf('\nMean test error is as follow:\n');
mean(errorPcs)

fprintf('\nStandard deviation in test error is as follow:\n');
std(errorPcs)
fprintf('\n');
