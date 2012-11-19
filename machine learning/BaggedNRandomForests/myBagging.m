%{
   CSci5525 Fall'12 Homework 3
   login: sharm163@umn.edu
   date: 11/17/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: learn bagging classifier for given data and cross ...
           validate it
%}


function[testError] = myBagging(dataFileName, numBaseClassifiers, ...
                                numFolds, numLayers)
                            
numBaseClassifiers = numBaseClassifiers(:);
                            
%load the data set
load(dataFileName);

%randomly shuffle data
randInd = randperm(size(data, 1));
permData = data(randInd, :);
permLabels = labels(randInd, :);

%get size of dataset
sizeData = size(permData,1);

%size of data to be left for validation
leftDataSize = int16(sizeData/numFolds);

%store mean error for each base classifier
baseClassifierTrainErrPcs = zeros(size(numBaseClassifiers, 1), 1);
baseClassifierTestErrPcs = zeros(size(numBaseClassifiers, 1), 1);

%learn bagged classifiers
for numBaseIter=1:size(numBaseClassifiers, 1)
    
    %number of base classifiers
    numBaseClassifier = numBaseClassifiers(numBaseIter);
    
    
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
    
        %learn the bagged Classifier and get the training and test error
        [trainErr, testErr] = bagging(trainingData, trainingLabels,...
                                    validationData, validationLabels, ...
                                    numBaseClassifier, numLayers);
    
        testErrorPcs(iter) = testErr;
        trainErrorPcs(iter) = trainErr;
    end

    fprintf('\n*************************************************\n');
    fprintf('\nNo. of Base Classifier: %d\n', numBaseClassifier);
    %trainErrorPcs
    fprintf('\nTrain Errors: \n');
    for iter=1:size(trainErrorPcs, 1)
        fprintf('%d\n', trainErrorPcs(iter));
    end
    
    meanTrainErr = mean(trainErrorPcs);
    stdTrainErr = std(trainErrorPcs);
    fprintf('\nMean train error is %d\n', meanTrainErr);
    fprintf('\nStandard deviation in train error is %d', stdTrainErr);
    fprintf('\n');
    
    %testErrorPcs
    fprintf('\nTest Errors: \n');
    for iter=1:size(testErrorPcs, 1)
        fprintf('%d\n', testErrorPcs(iter));
    end
    
    meanTestErr = mean(testErrorPcs);
    stdTestErr = std(testErrorPcs);
    fprintf('\nMean test error is %d', meanTestErr);
    fprintf('\nStandard deviation in test error is %d', stdTestErr);
    fprintf('\n');
    
    %store the learned mean error and test error
    baseClassifierTrainErrPcs(numBaseIter) = meanTrainErr;
    baseClassifierTestErrPcs(numBaseIter) = meanTestErr;
end

%set return value
testError = baseClassifierTrainErrPcs;

fprintf('\nNo. of Base Classifiers\tTrainError\tTestError');
for numBaseIter=1:size(numBaseClassifiers, 1)
    fprintf('\n%d\t%d\t%d', numBaseClassifiers(numBaseIter), ...
            baseClassifierTrainErrPcs(numBaseIter), ...
            baseClassifierTestErrPcs(numBaseIter));
    
end

%plot base classifiers vs training errors 
clf
plot(numBaseClassifiers, baseClassifierTrainErrPcs);
xlabel('No. of base classifiers');
ylabel('Training Error');
title('Bagging');
pause
clf
%plot base classifiers vs test errors
plot(numBaseClassifiers, baseClassifierTestErrPcs);
xlabel('No. of base classifiers');
ylabel('Test Error');
title('Bagging');
