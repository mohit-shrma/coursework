%{
   CSci5525 Fall'12 Homework 3
   login: sharm163@umn.edu
   date: 11/17/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: learn bagging classifier for given training data and ...
           apply it on test set data
%}

function [trainErrorPc, testErrorPc] = rForest(fileID,trainingData, trainingLabels,...
                                             testData, testLabels, ...
                                               sizeFeatureSet, ...
                                               depth)


    %size of taining data
    sizeTrainData = size(trainingData, 1);
    
    %get valid attributes list for data
    eligibleAttribs = [1:size(trainingData, 2)];
    
    %number of bootstrapped samples to learn classifiers, using 30 as default
    numBaseClassifier = 30;
    
    %store learned classifier in array
    baggedClassifiers = Node.empty(numBaseClassifier, 0);
    
    %learn base classifiers
    for baseClassIter=1:numBaseClassifier
        %build bootstrap data 
        [bootstrapData, origIdx] = datasample(trainingData, ...
                                              size(trainingData, 1)); 
        %get corresponding bootstrap labels
        bootstrapLabels = trainingLabels(origIdx);
        
        %learn binary split decision tree of given depth on bootstrapped data
        startDepth = 0;
        isWeighted = 0;
        dataWeights = ones(size(bootstrapData, 1), 1)/size(bootstrapData, 1);
        isRandomForest = 1;
        learnedTreeRoot = learnBinDtree(bootstrapData, bootstrapLabels, ...
                                        startDepth, depth, eligibleAttribs, ...
                                        mode(trainingLabels), ...
                                        isWeighted, dataWeights, ...
                                        isRandomForest, sizeFeatureSet);
        %store the learned decision tree
        baggedClassifiers(baseClassIter) = learnedTreeRoot;

    end
    
    
    %evaluate learned decision tree on training data
    errorCount = 0;
    for validIter=1:size(trainingData, 1)
        %get prediction from learn stump
        label = baggedPrediction(baggedClassifiers, trainingData(validIter, :));
        if label ~= trainingLabels(validIter)
            errorCount = errorCount + 1;
        end
    end
    trainErrorPc = errorCount/size(trainingData,1);
    %fprintf(fileID, '\nError rate on training set is %d', trainErrorPc);
    
    %evaluate learned decision tree on test data
    errorCount = 0;
    for validIter=1:size(testData, 1)
        %get prediction from learn stump
        label = baggedPrediction(baggedClassifiers, testData(validIter, :));
        if label ~= testLabels(validIter)
            errorCount = errorCount + 1;
        end
    end
    testErrorPc = errorCount/size(testData,1);
    %fprintf(fileID, '\nError rate on test set is %d', testErrorPc);
    



    
    