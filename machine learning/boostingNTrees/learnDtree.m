%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 10/25/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: learn decision tree on passed data and labels
%}

function [root] = learnDtree(data, labels, currDepth, ...
                             maxDepth, eligibleAttribs, ...
                             prevMajorityLabel, isWeighted, ...
                             dataWeights)

%get the number of attributes
numAttribs = size(data, 2);

%get the size of data
sizeData = size(data, 1);

%define all the node properties to default
%num of child, -> in case of leaf nodes will be 0
root.numChild = 0;
%class label assigned to node in case of leaf
root.classLabel = -99;
%store the current depth
root.depth = currDepth;
%initialize best atribute to current node in caase any found later
root.attrib = -99;
%similarly initialize child attribute value as default
root.childAttribVal = -99;
%initialize child to default empty
root.child = struct(root);

%compute sum of weights if weighted
netDataWeights = sum(dataWeights);    

if sizeData == 0 || currDepth == maxDepth
    %pass node with default label 
    root.numChild = 0;
    root.classLabel = prevMajorityLabel;
elseif length(unique(labels)) == 1
    %if all passed data of same label    
    root.numChild = 0;
    root.classLabel = labels(1);  
elseif isempty(eligibleAttribs)
    %if all the eligible attributes is empty or {}, return majority
    %value
    root.numChild = 0;
    root.classLabel = mode(labels);
else
    %find the best attribute to split on
    bestAttrib = -1;
    bestCondnEntropy = 1000;

    %search for the best attribute to split
    for attribIter=1:length(eligibleAttribs)
        currAttrib = eligibleAttribs(attribIter);
        attribValues = unique(data(:, currAttrib));
        %compute conditional entropy and class count corresponding to
        %each attribute
        conditionalEntropy = 0;
        for attribValIter=1:size(attribValues, 1)
            %get the indices having current attribute value
            dataInd = find(data(:, currAttrib) == ...
                           attribValues(attribValIter));
            if isWeighted ~= 1
                %compute entropy without considering weights
                %get the number for corresponding classes, assuming two
                %classes 1 & 2
                class1Count = length(find(labels(dataInd) == 1));
                class2Count = length(dataInd) - class1Count;
                conditionalEntropy = conditionalEntropy + ...
                    computeConditionalEntropy(sizeData, class1Count, ...
                                              class2Count); 
            else
                %compute entropy considering weights on data
                %get the corresponding row indices of class1 & class2
                class1Ind = find(labels(dataInd) == 1);
                class2Ind = find(labels(dataInd) == -1);
                class1Weights = sum(dataWeights(class1Ind));
                class2Weights = sum(dataWeights(class2Ind));
                conditionalEntropy = conditionalEntropy + ...
                    computeConditionalEntropy(netDataWeights, class1Weights, ...
                                              class2Weights); 
            end
        end
        
        if conditionalEntropy < bestCondnEntropy
            bestAttrib = currAttrib;
            bestCondnEntropy = conditionalEntropy;
        end
    end
    
    %assign best atribute to current node
    root.attrib = bestAttrib;
    
    %assign num of children to current node
    root.numChild = length(unique(data(:, bestAttrib)));
    
    %get majority label of currData
    majLabel = mode(labels);
    
    %passed the attributes filtering out the current attribute
    currAttribInd = find(eligibleAttribs == bestAttrib);
    filteredAttribs = [eligibleAttribs(1:currAttribInd-1) ...
                       eligibleAttribs(currAttribInd+1:length(eligibleAttribs))];
    
    
    %for each possible value of attribute get child node
    attribValues = unique(data(:, bestAttrib));
    for attribValIter=1:size(attribValues, 1)
        %out of the given data get data having current attribute
        %value
        filteredIndices = find(data(:, bestAttrib) == ...
                               attribValues(attribValIter));
        filteredData = data(filteredIndices, :);
        filteredLabels = labels(filteredIndices);
        filteredDataWeights = dataWeights[filteredIndices];
        %weird initialization to make the recursive substructure
        %assignment works. If don't do this for  then
        %report structure can not be assigned as they are different
        if attribValIter == 1
            root.child.child = struct([]);
        end
        %learn subtree for this attribute
        
        root.child(attribValIter) = learnDtree(filteredData, ...
                                               filteredLabels, ...
                                               currDepth + 1, ...
                                               maxDepth, ...
                                               filteredAttribs, ...
                                               majLabel, isWeighted, ...
                                               filteredDataWeights);
        root.childAttribVal(attribValIter) = attribValues(attribValIter);  
    end
    
    
end

