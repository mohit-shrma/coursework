%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: Fisher's linear discriminant
%}

function [] = fisher(dataFileName, numFolds)

%load given data
load(dataFileName);



%randomly shuffle data
randInd = randperm(size(data, 1));
permData = data(randInd, :);
permLabels = labels(randInd, :);


%get size of dataset
sizeData = size(permData,1);

%size of data to be left for validation
leftDataSize = int16(sizeData/numFolds);

%store learned parameters for each validation
bestLearnedProjMean = [];
bestLearnedSharedCvariance = [];
bestLearnedClassPriors = [];
bestLearnedWeightVec = [];


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
    
    %learn parameters by training
    flagWithinI = 0;
    [projectedMeans, sharedCovariance, classPriors, weightVec] = ...
        fisherTrain(trainingData, trainingLabels, flagWithinI);
    
    
    %evaluate parameters on validation data
    errorCount = 0;
    for validIter=1:size(validationData,1)
        projectedDataRow = (weightVec' * validationData(validIter, :)')';
        [posteriorVec, maxLabel] = softMax(projectedDataRow, ...
                                           projectedMeans, ...
                                           sharedCovariance, ...
                                           classPriors);
        if maxLabel ~= validationLabels(validIter)
            errorCount = errorCount + 1;
        end
    end
    
    currErrorPc = errorCount/size(validationData,1);
    
    if currErrorPc <= max(errorPcs)
        bestLearnedProjMean = projectedMeans;
        bestLearnedSharedCvariance = sharedCovariance;
        bestLearnedClassPriors = classPriors;    
        bestLearnedWeightVec = weightVec;
    end
    
    errorPcs(iter) = errorCount/size(validationData,1);
    
end
fprintf(dataFileName);
%errorPcs
fprintf('\nmean error is as follow:\n');
mean(errorPcs)

fprintf('\nstandard deviation in error is as follow:\n');
std(errorPcs)
%bestLearnedProjMean
%bestLearnedSharedCvariance
%bestLearnedClassPriors
%bestLearnedWeightVec
