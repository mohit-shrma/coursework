%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 10/25/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: learn decision tree on training data, print it and ...
       output error rate on training set and test set 
%}

function [trainErrorPc, testErrorPc] = dtree(trainingData, trainingLabels,...
                                             testData, testLabels, ...
                                             depth)
    %size of taining data
    sizeTrainData = size(trainingData, 1);
    
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
      
    %learn decision tree of passed depth on current training data    
    startDepth = 0;
    isWeighted = 0;
    dataWeights = ones(sizeTrainData, 1)/sizeTrainData;
    learnedTreeRoot = learnDtree(trainingData, trainingLabels, ...
                                 startDepth, depth, eligibleAttribs, ...
                                 mode(trainingLabels), isWeighted, ...
                                 dataWeights);
        
    %print the learned decision tree
    fprintf('\n\n************Decision tree is as follow **************\n')
    printDtree(learnedTreeRoot);
    
    %evaluate learned decision tree on training data
    errorCount = 0;
    for validIter=1:size(trainingData, 1)
        %get prediction from learn stump
        label = predictFrmDtree(trainingData(validIter, :), learnedTreeRoot);
        if label ~= trainingLabels(validIter)
            errorCount = errorCount + 1;
        end
    end
    trainErrorPc = errorCount/size(trainingData,1);
    fprintf('\nError rate on training set is %d', trainErrorPc);
    
    %evaluate learned decision tree on test data
    errorCount = 0;
    for validIter=1:size(testData, 1)
        %get prediction from learn stump
        label = predictFrmDtree(testData(validIter, :), learnedTreeRoot);
        if label ~= testLabels(validIter)
            errorCount = errorCount + 1;
        end
    end
    testErrorPc = errorCount/size(testData,1);
    fprintf('\nError rate on test set is %d', testErrorPc);
