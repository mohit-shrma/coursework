%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: apply naive bayes with multivariate gaussian and use ...
       100 random 80-20 train test split to evaluate
%}



%load given data
%TODO: read from commandline, the data file and then load it
dataFileName = 'Pima.mat';
load(dataFileName);

%get number of runs
%TODO: read from commandline, the data file and then load it
runs = 100;

%training-set percentage vector
%TODO: read from commandline, the data file and then load it
trainPcVec = [5 10 15 20 25 35 40 50 60 75 90];

%get size of dataset
sizeData = size(data,1);

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
        [gaussianParams, classPriors] = naiveBayesTrain(trainData, ...
                                                        trainLabels);
        
        %start cross-validation
        errorCount = 0;
        for validIter=1:size(valData, 1)
            predLabel = naiveBayesPredict(classes, valData(validIter,:), ...
                                      gaussianParams, classPriors);
            if predLabel ~= valLabels(validIter)
                errorCount = errorCount + 1;
            end
        end
        errorPc = errorCount/size(valData, 1);
        errorPcRuns(iterRun, iterTrain) = errorPc;
    end
end

meanErr = mean(errorPcRuns)
stdErr = std(errorPcRuns)

