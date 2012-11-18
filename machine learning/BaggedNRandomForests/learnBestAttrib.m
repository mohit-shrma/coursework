%{
   CSci5525 Fall'12 Homework 3
   login: sharm163@umn.edu
   date: 11/17/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: learn best attribute to make a binary split, and value ...
       of that attribute to make binary split
%}

function [bestAttrib, bestAttribVal] = learnBestAttrib(data, labels,...
                                              eligibleAttribs,...
                                              isWeighted, dataWeights)

%get the number of attributes
numAttribs = size(data, 2);

%get the size of data
sizeData = size(data, 1);

%search for best attribute to split
bestAttrib = -1;
bestAttribVal = -979;
bestCondnEntropy = 50000;

for attribIter=1:length(eligibleAttribs)
    currAttrib = eligibleAttribs(attribIter);
    %get the best current attribute value to split and the
    %conditional entropy
    [condEntropy, bestCurrAttribVal] = getBestAttribVal(data(:, currAttrib), ...
                                                    labels, isWeighted,...
                                                    dataWeights);
    if condEntropy < bestCondnEntropy
        bestAttrib = currAttrib;
        bestAttribVal = bestCurrAttribVal;
        bestCondnEntropy = condEntropy;
    end
    
end


%{
find the best attribute value to split on for the given attribute ...
        values of the data
%}
function [bestCondEntropy, bestSplitVal] = getBestAttribVal(dataAttribVal,...
                                                labels, isWeighted,...
                                                dataWeights)

%get the size of data
sizeData = size(dataAttribVal, 1);

%sort the attribute values
[sortedDataAttribVal, oldIndices] = sort(dataAttribVal);

%build the corresponding sorted labels
sortedLabels = labels(oldIndices);

%build the corresponding sorted weights
sortedWeights = dataWeights(oldIndices);

%get the number for corresponding classes, assuming two
%classes 1 & 2
class1Count = length(find(sortedLabels == 1));
class2Count = length(find(sortedLabels ~= 1));


%get the corresponding row indices of class1 & class2
class1Ind = find(labels == 1);
class2Ind = find(labels ~= 1);
class1Weights = sum(dataWeights(class1Ind));
class2Weights = sum(dataWeights(class2Ind));
netDataWeights = sum(dataWeights);

%best split value
bestSplitVal = -979;
bestCondEntropy = 1000;

%store the class count for attrib val < split val
LessClass1Count = 0;
LessClass2Count = 0;

%store the class count for attrib val >= split val
Gr8EqClass1Count = 0;
Gr8EqClass2Count = 0;

%first attrib sorted value no prior attrib values, can directly
%conclude num. of all < than attribVal = 0
%num. of all >= than attribVal = orig. class count
LessClass1Count = 0;
LessClass2Count = 0;
LessClass1Weight = 0;
LessClass2Weight = 0;
Gr8EqClass1Count = class1Count;
Gr8EqClass2Count = class2Count;
Gr8EqClass1Weight = class1Weights;
Gr8EqClass2Weight = class2Weights;
bestSplitVal = sortedDataAttribVal(1);
if ~isWeighted
    bestCondEntropy = computeConditionalEntropy(sizeData, LessClass1Count, ...
                                            LessClass2Count) + ...
        computeConditionalEntropy(sizeData, Gr8EqClass1Count, ...
                              Gr8EqClass2Count);
else
    bestCondEntropy = computeConditionalEntropy(netDataWeights, LessClass1Weight, ...
                                            LessClass2Weight) + ...
        computeConditionalEntropy(netDataWeights, Gr8EqClass1Weight, ...
                              Gr8EqClass2Weight);
end



%for each attribute value decide on best split value by splitting
%on middle of two consecutive attribute in sorted attributes
dataIter = 1;
lastSplitAttribVal = sortedDataAttribVal(1);
for iter=2:size(sortedDataAttribVal)
    splitAttribVal = (sortedDataAttribVal(iter) + ...
                      sortedDataAttribVal(iter-1)) / 2;
    if lastSplitAttribVal == splitAttribVal
        %if last splitted attribute value equals current split
        %value then continue
        continue;
    end
    
    %update the class counts with each data satisfying the class criteria
    while sortedDataAttribVal(dataIter) < splitAttribVal && dataIter ...
            <= sizeData
        if sortedLabels(dataIter) == 1
            LessClass1Count = LessClass1Count + 1;
            LessClass1Weight = LessClass1Weight + sortedWeights(dataIter);
            Gr8EqClass1Count = Gr8EqClass1Count - 1;
            Gr8EqClass1Weight = Gr8EqClass1Weight - sortedWeights(dataIter);
        else
            LessClass2Count = LessClass2Count + 1;
            LessClass2Weight = LessClass2Weight + sortedWeights(dataIter);
            Gr8EqClass2Count = Gr8EqClass2Count - 1;
            Gr8EqClass2Weight = Gr8EqClass2Weight - sortedWeights(dataIter);
        end
        dataIter = dataIter + 1;
    end
    
    %compute conditional entropy for current split
    if ~isWeighted
        condnEntropy = computeConditionalEntropy(sizeData, LessClass1Count, ...
                                                LessClass2Count) + ...
                          computeConditionalEntropy(sizeData, Gr8EqClass1Count, ...
                                                Gr8EqClass2Count);
    else
        condnEntropy = computeConditionalEntropy(netDataWeights, LessClass1Weight, ...
                                            LessClass2Weight) + ...
                          computeConditionalEntropy(netDataWeights, Gr8EqClass1Weight, ...
                                            Gr8EqClass2Weight);
    end
   
    
    %update best values if current split best
    if condnEntropy < bestCondEntropy
        bestCondEntropy = condnEntropy;
        bestSplitVal = splitAttribVal;
    end
    
end


