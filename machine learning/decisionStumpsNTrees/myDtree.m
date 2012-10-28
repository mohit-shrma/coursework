%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 10/25/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: learn decision tree on training data for passed depth and apply k-fold ...
       validation where k is passed as an argument to function
%}

function [] = myDtree(dataFileName, depth, numFolds)

load(dataFileName);

%as data is skewed randomly shuffle data
randInd = randperm(size(data, 1));
permData = data(randInd, :);
permLabels = labels(randInd, :);

%get size of dataset
sizeData = size(permData,1);

%size of data to be left for validation
leftDataSize = int16(sizeData/numFolds);

%store accuracy of learned parameters for each validation
errorPcs = zeros(numFolds, 1);

bestLearnedTree = []

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
    

    %get valid attributes list for data
    eligibleAttribs = [1:size(data, 2)];
    
    %check if it's mushroom dataset then take out feature 11
    filteredOutAttrib = 11;
    if dataFileName == 'Mushroom.mat'
        eligibleAttribs = [eligibleAttribs(1:filteredOutAttrib-1) ...
                           eligibleAttribs(filteredOutAttrib+1:end)];
    end
    
    %learn decision tree of passed depth on current training data    
    startDepth = 0;
    learnedTreeRoot = learnDtree(trainingData, trainingLabels, ...
                                 startDepth, depth, eligibleAttribs, ...
                                 mode(trainingLabels));
    
    %evaluate learned decision tree on validation data
    errorCount = 0;
    for validIter=1:size(validationData,1)
        %get prediction from learn stump
        label = predictFrmDtree(validationData(validIter, :), learnedTreeRoot);
        if label ~= validationLabels(validIter)
            errorCount = errorCount + 1;
        end
    end
    
    currErrorPc = errorCount/size(validationData,1);
    
    if currErrorPc <= max(errorPcs)
        bestLearnedTree = learnedTreeRoot;
    end
    
    errorPcs(iter) = errorCount/size(validationData,1);

end