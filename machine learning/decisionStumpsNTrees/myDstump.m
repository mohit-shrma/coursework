%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 10/25/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: learn decision stumps on training data and apply k-fold ...
       validation where k is passed as an argument to function
%}

function [] = diagFisher(dataFileName, numFolds)

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

bestLearnedAttrib = -1;
bestLearnedAttribValueClass = [];
        
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
    
    %learn decision stump on current training data
    %TODO: write learnDstump method
    [stumpAttrib, stumpAttribValueClass] = learnDstump(trainingdata, trainingLabels);
    
    %evaluate learned decision stump on validation data
    errorCount = 0;
    for validIter=1:size(validationData,1)
        %get prediction from learn stump
        %TODO: write predict from stump method
        label = predictFromStump(validationData(validIter, :), ...
                                 stumpAttrib, stumpAttribValueClass);
        if label ~= validationLabels(validIter)
            errorCount = errorCount + 1;
        end
    end
    
    currErrorPc = errorCount/size(validationData,1);
    
    if currErrorPc <= max(errorPcs)
        bestLearnedAttrib = stumpAttrib;
        bestLearnedAttribValueClass = stumpAttribValueClass;
    end
    
    errorPcs(iter) = errorCount/size(validationData,1);

end


fprintf(dataFileName);
%errorPcs
fprintf('\nmean error is as follow:\n');
mean(errorPcs)

fprintf('\nstandard deviation in error is as follow:\n');
std(errorPcs)


%predict label from learned stump
function[label] = predictFromStump(dataVec, stumpAttrib, ...
                                   stumpAttribValueClass)

dataStumpAttribVal = dataVec[stumpAttrib]
classRowInd = find(dataVec(stumpAttrib) == dataStumpAttribVal)
%TOSO: check if empty then assign default class ?
label = stumpAttribValueClass(classInd, 2)
