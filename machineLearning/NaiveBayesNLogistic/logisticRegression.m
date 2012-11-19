%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: apply logistic regression and use ...
       100 random 80-20 train test split to evaluate
%}

function [] = logisticRegression(dataFileName, trainPcVec)

%load given data
load(dataFileName);

%get number of runs
runs = 100;

%get size of dataset
sizeData = size(data,1);

%normalize data
%data = meanNormalizeData(data);

%get classes
classes = unique(labels);

%divide into training data and validation data
trainDataSize = int16(0.8 * sizeData);

%error percent vector across all runs
errorPcRuns = zeros(runs, size(trainPcVec, 2));

%mean error percent vector across all runs
meanErr = zeros(size(trainPcVec, 2));

%std dev across all runs
stdErr = zeros(size(trainPcVec, 2));


for iterRun=1:runs
    permInd = randperm(sizeData)';
    
    trainDataInd = permInd(1:trainDataSize);
    valDataInd = permInd(trainDataSize+1:sizeData);
    
    valData = data(valDataInd, :);
    valLabels = labels(valDataInd, :); 
    
    for iterTrain=1:size(trainPcVec, 2)
        currTrainDataSize = int16(trainDataSize*((trainPcVec(iterTrain)/100)));
        
        trainData = data(trainDataInd(1:currTrainDataSize), :);
        trainLabels = labels(trainDataInd(1:currTrainDataSize), :); 
        
        %train using the training data
        %use one vs all training to learn weights for each class
        weights = oneVsAllTrain(trainData, trainLabels);
        
        %start cross-validation
        errorCount = 0;
        for validIter=1:size(valData, 1)
            %add bias to data
            newValData = [1 valData(validIter, :)];
            
            %apply sigmoid on data learned for each class
            sigmoidClassesVec = weights'*newValData';
            
            %predict the class having max value as the label
            [maxVal, maxInd] = max(sigmoidClassesVec);
            predLabel = maxInd;
           
            if predLabel ~= valLabels(validIter)
                errorCount = errorCount + 1;
            end
        end
        
        errorPc = errorCount/size(valData, 1);
        errorPcRuns(iterRun, iterTrain) = errorPc;
    end
end
fprintf('\nmean error is as follow:\n');
meanErr = mean(errorPcRuns)
fprintf('\nstandard dev is as follow:\n');
stdErr = std(errorPcRuns)
errorbar(meanErr, stdErr, 'r');
        