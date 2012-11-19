%{
   CSci5525 Fall'12 Homework 3
   login: sharm163@umn.edu
   date: 11/17/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: learn binary decision tre for given data
%}

function [root] = learnBinDtree(data, labels, currDepth, ...
                             maxDepth, eligibleAttribs, ...
                             prevMajorityLabel, isWeighted, ...
                                dataWeights, isRandomForest, ...
                                sizeFeatureSet)
    
%original attributes
origAttribs = eligibleAttribs;    
    

%get the size of data
sizeData = size(data, 1);

%initialize root to be empty
root = [];

if sizeData == 0 
    %pass node with default label
    root = Node(-1, -1, prevMajorityLabel);
elseif length(unique(labels)) == 1
    %if all passed data of same label
    root = Node(-1, -1, labels(1));
elseif isempty(eligibleAttribs) || currDepth == maxDepth
    %TODO: depth check
    %if all the eligible attributes is empty or {} or currentDepth
    %equals maxDepth, return majority value
    root = Node(-1, -1, mode(labels));
else
    
    if isRandomForest
        %randomly choose the eligible attributes of specified size
        %y = datasample(1:10,5,'Replace',false)
        eligibleAttribs = datasample(origAttribs, sizeFeatureSet, ...
                                     'Replace', false);
    end
    
    
    
    %find the best attribute to split on
    [bestAttrib, bestAttribVal] = learnBestAttrib(data, labels, ...
                                                  eligibleAttribs, ...
                                                  isWeighted, dataWeights);
    
    %initialize current node with these learned attribute and value
    root = Node(bestAttrib, bestAttribVal, -1);
    
    %get majority label of currData
    majLabel = mode(labels);
    
    if isRandomForest
        %give original attributes
        filteredAttribs = origAttribs;
    else
        %give filteredattributes
        %passed the attributes filtering out the current attribute
        currAttribInd = find(eligibleAttribs == bestAttrib);
        filteredAttribs = [eligibleAttribs(1:currAttribInd-1) ...
                           eligibleAttribs(currAttribInd+1:length(eligibleAttribs))];
    end
    
    %get left child node of current attribute
    %out of the given data get data having < current attribute
    %split value
    filteredLeftIndices = find(data(:, bestAttrib) < bestAttribVal);
    filteredLeftData = data(filteredLeftIndices, :);
    filteredLeftLabels = labels(filteredLeftIndices);
    filteredLeftDataWeights = dataWeights(filteredLeftIndices);
    leftChild = learnBinDtree(filteredLeftData, filteredLeftLabels, currDepth ...
                           + 1, maxDepth, filteredAttribs, ...
                           majLabel, isWeighted, filteredLeftDataWeights, ...
                              isRandomForest, sizeFeatureSet);
    
    %get right child node of current attribute
    filteredRightIndices = setdiff([1:sizeData], filteredLeftIndices);
    filteredRightData = data(filteredRightIndices, :);
    filteredRightLabels = labels(filteredRightIndices);
    filteredRightDataWeights = dataWeights(filteredRightIndices);
    rightChild = learnBinDtree(filteredRightData, filteredRightLabels, currDepth ...
                           + 1, maxDepth, filteredAttribs, ...
                           majLabel, isWeighted, filteredRightDataWeights, ...
                               isRandomForest, sizeFeatureSet);
    
    %add left child to root
    root.addLeftChild(leftChild);
    %add right child to root
    root.addRightChild(rightChild);
end
