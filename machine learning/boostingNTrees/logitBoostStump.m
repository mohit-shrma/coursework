%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 10/25/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: apply logitboost using stumps on trianing data, print it and ...
       output error rate on training set and test set 
%}

function [trainErrorPc, testErrorPc] = logitBoostStump(trainingData, trainingLabels,...
                                                  testData, testLabels, ...
                                                  numStumps)

    %get valid attributes list for data
    eligibleAttribs = [1:size(trainingData, 2)];
    
    %check if it's mushroom dataset then take out feature 11
    %bad code for the sake of this question, remove below line will
    %work for any data set
    dataFileName = 'Mushroom.mat';
    filteredOutAttrib = 11;
    if strcmp(dataFileName, 'Mushroom.mat') == 1
        eligibleAttribs = [eligibleAttribs(1:filteredOutAttrib-1) ...
                           eligibleAttribs(filteredOutAttrib+1:end)];
    end
    
    %size of taining data
    sizeTrainData = size(trainingData, 1);
    
    %initialize weights vector for data
    weightsTrainingData = ones(sizeTrainData, 1)/sizeTrainData;
    
    %initialize boost iteration weights
    boostClassifierWeights = zeros(numStumps, 1);
    
    %store learned stumps
    learnedStumps = [];
    
    %we are learning weighted stumps
    isWeighted = 1;
    
    %weights on data
    dataWeights = ones(sizeTrainData, 1)/sizeTrainData;
    
    
    
    %run boosting iterations
    for boostIter=1:numStumps
        
        %learn weighted decision stump
        startDepth = 0;
        %decision stump height is 1
        depth = 1;
        learnedDStump = learnDtree(trainingData, trainingLabels, ...
                                          startDepth, depth, ...
                                           eligibleAttribs, ...
                                           mode(trainingLabels), ...
                                           isWeighted, dataWeights);
        %printDtree(learnedDStump);
                                       
        if size(learnedStumps, 1) == 0
            
            for iter=1:numStumps
                learnedStumps = [struct(learnedDStump); learnedStumps];
            end
        end
        
        learnedStumps(boostIter) = learnedDStump;
        %evaluate learned decision stump on training data
        errorCount = 0;
        indicatorError = ones(sizeTrainData, 1);
        weightedErr = 0;
        
        for trainIter=1:size(trainingData, 1)
            %get prediction from learned stump
            predLabel = predictFrmDtree(trainingData(trainIter, :), learnedDStump);
            if predLabel ~= trainingLabels(trainIter)
                indicatorError(trainIter) = -1;
                errorCount = errorCount + 1;
                weightedErr = weightedErr + weightsTrainingData(trainIter);
            end
        end
        
        
        
        %eps: error
        %normalize the weighted err, (epsM)
        eps = weightedErr / sum(weightsTrainingData);
        %compute current classifier weight
        alpha = (log((1-eps)/eps));
        %reweight the data
        weightsTrainingData = ((weightsTrainingData) + log(1+exp(-alpha*indicatorError)))/(sum(weightsTrainingData));
        boostClassifierWeights(boostIter) = alpha;
    end
    
    learnedStumps = learnedStumps(1:boostIter);
    
    %make predictions using final model, using predictFrmBoost method
    %evaluate training data from new model
    errorCount = 0;
    for validIter=1:size(trainingData, 1)
        %get prediction from learned boosted model
        label = predictFrmBoost(trainingData(validIter, :), ...
                                learnedStumps, boostClassifierWeights);
        if label ~= trainingLabels(validIter)
            errorCount = errorCount + 1;
        end
    end
    trainErrorPc = errorCount/size(trainingData,1);
    fprintf('\nError rate on training set is %d', trainErrorPc);
    
    %evaluate new model on test data
    errorCount = 0;
    for validIter=1:size(testData, 1)
        %get prediction from learned boosted model
        label = predictFrmBoost(testData(validIter, :), learnedStumps, ...
                                boostClassifierWeights);
        if label ~= testLabels(validIter)
            errorCount = errorCount + 1;
        end
    end
    testErrorPc = errorCount/size(testData,1);
    fprintf('\nError rate on test set is %d', testErrorPc);
       
       