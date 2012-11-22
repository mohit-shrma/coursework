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
%strcat(num2str(numLayers), 'bagErr.png')    
%open file to write or log output    
fileID = fopen(strcat(num2str(numLayers), 'bagLog.txt'),'w');    
    
%columnize num of base classifiers sequence                            
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
        [trainErr, testErr] = bagging(fileID, trainingData, trainingLabels,...
                                    validationData, validationLabels, ...
                                    numBaseClassifier, numLayers);
    
        testErrorPcs(iter) = testErr;
        trainErrorPcs(iter) = trainErr;
    end

    fprintf(fileID, '\n*************************************************\n');
    fprintf(fileID, '\nNo. of Base Classifier: %d\n', numBaseClassifier);
    
    %trainErrorPcs & test Error
    fprintf(fileID, '\nTrain Errors\tTest Errors: \n');
    for iter=1:numFolds
        fprintf(fileID, '%d\t%d\n', trainErrorPcs(iter), testErrorPcs(iter));
    end
    
    meanTrainErr = mean(trainErrorPcs);
    stdTrainErr = std(trainErrorPcs);
    fprintf(fileID, '\nMean train error is %d', meanTrainErr);
    fprintf(fileID, '\nStandard deviation in train error is %d\n', stdTrainErr);
    
    meanTestErr = mean(testErrorPcs);
    stdTestErr = std(testErrorPcs);
    fprintf(fileID, '\nMean test error is %d', meanTestErr);
    fprintf(fileID, '\nStandard deviation in test error is %d', stdTestErr);
    fprintf(fileID, '\n');
    
    %store the learned mean error and test error
    baseClassifierTrainErrPcs(numBaseIter) = meanTrainErr;
    baseClassifierTestErrPcs(numBaseIter) = meanTestErr;
end

%set return value
testError = baseClassifierTrainErrPcs;

fprintf(fileID, '\n********* All mean train & test Errors *********\n');

fprintf(fileID, '\nNo. of Base Classifiers\tTrainError\tTestError');
for numBaseIter=1:size(numBaseClassifiers, 1)
    fprintf(fileID, '\n%d\t%d\t%d', numBaseClassifiers(numBaseIter), ...
            baseClassifierTrainErrPcs(numBaseIter), ...
            baseClassifierTestErrPcs(numBaseIter));
    
end

%plot base classifiers vs training errors 
h = figure;
errorbar(numBaseClassifiers, baseClassifierTrainErrPcs);
xlabel('No. of base classifiers');
ylabel('Error');
title(strcat(num2str(numLayers),'-layer Bagging'));
%saveas(h, strcat(num2str(numLayers), 'bagTrain.png'), 'png');

%plot base classifiers vs test errors
%h = figure;
hold on;
errorbar(numBaseClassifiers, baseClassifierTestErrPcs, 'r');
%xlabel('No. of base classifiers');
%ylabel('Error');
%title(strcat(num2str(numLayers),'-layer Bagging'));
legend('Train', 'Test');
saveas(h, strcat(num2str(numLayers), 'bag.png'), 'png');

fclose(fileID);
