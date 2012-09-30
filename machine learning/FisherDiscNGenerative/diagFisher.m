%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: Fisher's linear discriminant
%}

%load given data
%TODO: read from commandline, the data file and then load it
dataFileName = 'iris.mat';
load(dataFileName);

%get number of folds for cross-validation
%TODO: 
numFolds = 10;

%get size of dataset
sizeData = size(data,1);

%size of data to be left for validation
leftDataSize = int16(sizeData/numFolds);

%store learned parameters for each validation
%TODO:hard coded num of learned parmas, do away with it 
learnedParams = zeros(numFolds, 3);
bestLearnedProjMean = [];
bestLearnedSharedCvariance = [];
bestLearnedClassPriors = [];
bestLearnedWeightVec = [];


%store accuracy of learned parameters for each validation
validationAccuracy = zeros(numFolds, 1);


for iter=1:numFolds
    %computation validation data range
    fprintf('\n validation number: %d', iter);
    validStart = (iter-1) * leftDataSize + 1;
    validEnd = validStart + leftDataSize - 1;
    if validEnd > sizeData
        validEnd = sizeData;
    end
    
    validationData = data(validStart:validEnd, :);
    validationLabels = labels(validStart:validEnd, :);
    
    if validEnd ~= sizeData 
        trainingData = [data(1:validStart-1, :); data(validEnd+1: ...
                                                   sizeData, :)];       
        trainingLabels = [labels(1:validStart-1); labels(validEnd+1: ...
                                                   sizeData, :)];
    else
        trainingData = data(1:validStart-1, :);       
        trainingLabels = labels(1:validStart-1);       
    end
    
    %learn parameters by training
    [projectedMeans, sharedCovariance, classPriors, weightVec] = ...
        fisherTrain(trainingData, trainingLabels);
    %learnedParams(iter, :) = [projectedMeans sharedCovariance ...
    %                    classPriors];
    
    
    
    %evaluate parameters on validation data
    trueLabelCount = 0;
    for validIter=1:size(validationData,1)
        projectedDataRow = (weightVec' * validationData(validIter, :)')';
        [posteriorVec, maxLabel] = softMax(projectedDataRow, ...
                                           projectedMeans, ...
                                           sharedCovariance, ...
                                           classPriors);
        if maxLabel == validationLabels(validIter)
            trueLabelCount = trueLabelCount + 1;
        end
    end
    
    currAccuracy = trueLabelCount/size(validationData,1);
    
    if currAccuracy >= max(validationAccuracy)
        bestLearnedProjMean = projectedMeans;
        bestLearnedSharedCvariance = sharedCovariance;
        bestLearnedClassPriors = classPriors;    
        bestLearnedWeightVec = weightVec;
    end
    
    validationAccuracy(iter) = trueLabelCount/size(validationData,1);
    
end

validationAccuracy
bestLearnedProjMean
bestLearnedSharedCvariance
bestLearnedClassPriors
bestLearnedWeightVec




