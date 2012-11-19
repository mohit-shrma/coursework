%{
   CSci5525 Fall'12 Homework 3
   login: sharm163@umn.edu
   date: 11/17/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: learn random forest classifier for given data and cross ...
           validate it
%}


function[testError] = myRForest(dataFileName, randFeatureSetSizes, ...
                                numFolds, numLayers)

randFeatureSetSizes = randFeatureSetSizes(:);                            
                            
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

%store mean error for each random feature set
randFeatureTrainErrPcs = zeros(size(randFeatureSetSizes, 1), 1);
randFeatureTestErrPcs = zeros(size(randFeatureSetSizes, 1), 1);

for randFeatureSetIter=1:size(randFeatureSetSizes, 1)
    
    %size of fetaure set to be used to decide
    sizeFeatureSet = randFeatureSetSizes(randFeatureSetIter);    
    
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
    
        %learn the random forest Classifier and get the training
        %and test error
        %TODO: write rForest
        [trainErr, testErr] = rForest(trainingData, trainingLabels,...
                                    validationData, validationLabels, ...
                                    sizeFeatureSet, numLayers);
    
        testErrorPcs(iter) = testErr;
        trainErrorPcs(iter) = trainErr;
    end

    fprintf('\n*************************************************\n');
    fprintf('\nNo. of Base Classifier: %d\n', sizeFeatureSet);
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
    randFeatureTrainErrPcs(randFeatureSetIter) = meanTrainErr;
    randFeatureTestErrPcs(randFeatureSetIter) = meanTestErr;
end

%set return value
testError = randFeatureTestErrPcs;

fprintf('\nSize of Features\tTrainError\tTestError');
for randFeatureSetIter=1:size(randFeatureSetSizes, 1)
    fprintf('\n%d\t%d\t%d', randFeatureSetSizes(randFeatureSetIter), ...
            randFeatureTrainErrPcs(randFeatureSetIter), ...
            randFeatureTestErrPcs(randFeatureSetIter));
    
end

%plot FeatureSetSizes vs training errors 
h = figure;
plot(randFeatureSetSizes, randFeatureTrainErrPcs);
xlabel('Size of Features');
ylabel('Training Error');
title('Random Forests');
saveas(h, strcat(num2str(numLayers),'rfTrain.png'), 'png');
%plot FeatureSetSizes vs test errors
h = figure;
plot(randFeatureSetSizes, randFeatureTestErrPcs);
xlabel('Size of Features');
ylabel('Test Error');
title('Random Forests');
saveas(h, strcat(num2str(numLayers),'rfTest.png'), 'png');
