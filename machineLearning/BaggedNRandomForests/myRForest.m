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
%strcat(num2str(numLayers), 'bagErr.png')    
%open file to write or log output    
fileID = fopen(strcat(num2str(numLayers), 'forestLog.txt'),'w');    

%vectorize the feature set sizes
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
        [trainErr, testErr] = rForest(fileID, trainingData, trainingLabels,...
                                    validationData, validationLabels, ...
                                    sizeFeatureSet, numLayers);
    
        testErrorPcs(iter) = testErr;
        trainErrorPcs(iter) = trainErr;
    end

    fprintf(fileID, '\n*************************************************\n');
    fprintf(fileID, '\nSize of Feature Set: %d\n', sizeFeatureSet);
    
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
    randFeatureTrainErrPcs(randFeatureSetIter) = meanTrainErr;
    randFeatureTestErrPcs(randFeatureSetIter) = meanTestErr;
end

%set return value
testError = randFeatureTestErrPcs;

fprintf(fileID, '\n********* All mean train & test Errors *********\n');

fprintf(fileID, '\nSize of Features\tTrainError\tTestError');
for randFeatureSetIter=1:size(randFeatureSetSizes, 1)
    fprintf(fileID, '\n%d\t%d\t%d', randFeatureSetSizes(randFeatureSetIter), ...
            randFeatureTrainErrPcs(randFeatureSetIter), ...
            randFeatureTestErrPcs(randFeatureSetIter));
    
end

%plot FeatureSetSizes vs training errors 
h = figure;
plot(randFeatureSetSizes, randFeatureTrainErrPcs);
xlabel('Size of Features');
ylabel('Error');
title(strcat(num2str(numLayers),'-layer Random Forests'));
%saveas(h, strcat(num2str(numLayers),'rfTrain.png'), 'png');

%plot FeatureSetSizes vs test errors
%h = figure;
hold on;
plot(randFeatureSetSizes, randFeatureTestErrPcs), 'r';
%xlabel('Size of Features');
%ylabel('Test Error');
%title(strcat(num2str(numLayers),'-layer Random Forests'));
legend('Train', 'Test');
saveas(h, strcat(num2str(numLayers),'forest.png'), 'png');

fclose(fileID);
