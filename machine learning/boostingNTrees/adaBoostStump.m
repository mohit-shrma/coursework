%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 10/25/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: apply adaboost using stumps on trianing data, print it and ...
       output error rate on training set and test set 
%}

function [trainErrorPc, testErrorPc] = adaBoostStump(trainingData, trainingLabels,...
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
    
    sizeTrainData = size(trainingData, 1);
    
    %initialize weights vector for data
    weightsTrainingData = ones(sizeTrainData, 1)/sizeTrainData;
    
    %initialize boost iteration weights
    boostClassifierWeights = zeros(numstumps, 1)
    
    %store learned stumps
    learnedStumps = zeros(numStumps, 1);
    
    %run boosting iterations
    for boostIter=1:numStumps
        
        %learn weighted decision stump
        %TODO: write learn weighted dstump
        learnedDStump = learnWeightedStump(trainingData, trainingLabels, ...
                                          eligibleAttribs);
        learnedStumps(boostIter) = learnedDStump;
        %evaluate learned decision stump on training data
        errorCount = 0;
        weightedErr = 0;
        indicatorError = zeros(size(trainingData, 1), 1);
        for trainIter=1:size(trainingData, 1)
            %get prediction from learned stump
            %TODO: write predict Frm Dstump
            label = predictFrmDstump(trainingData(trainIter, :), learnedDStump);
            if label ~= trainingLabels(trainIter)
                weightedErr += weightsTrainingData(boostIter)
                indicatorError(trainIter) = 1;
            end
        end
        
        %normalize the weighted err, (epsM)
        normalizeWeightedErr = weightedErr / ...
            sum(weightsTrainingData)
        
        %find classifier weight
        boostClassifierWeights(boostIter) = log((1-normalizeWeightedErr)/normalizeWeightedErr);
        
        %update the weights
        weightsTrainingData = weightsTrainingData .* exp(alpha*indicatorError);
    end
    
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
