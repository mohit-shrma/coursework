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
                             prevMajorityLabel)

%get the number of attributes
numAttribs = size(data, 2);

%get the size of data
sizeData = size(data, 1);

%store the current depth
root.depth = currDepth;

if sizeData == 0 || currDepth == maxDepth
    %pass node with default label 
    root.numChild = 0;
    root.classLabel = prevMajorityLabel;
elseif length(unique(labels)) == 1
    %if all passed data of same label    
    root.numChild = 0;
    root.classLabel = labels(0);  
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
        attribValueClass = zeros(length(attribValues), 2);
        %compute conditional entropy and class count corresponding to
        %each attribute
        conditionalEntropy = 0;
        for attribValIter=1:size(attribValues, 1)
            %get the indices having current attribute value
            dataInd = find(data(:, currAttrib) == ...
                           attribValues(attribValIter));
            %get the number for corresponding classes, assuming two
            %classes 1 & 2
            class1Count = length(find(labels(dataInd) == 1));
            class2Count = length(dataInd) - class1Count;
            conditionalEntropy = conditionalEntropy + ...
                computeConditionalEntropy(sizeData, class1Count, ...
                                          class2Count); 
            %fill the corresponding class in attribute-class matrix
            attribValueClass(attribValIter, 1) = ...
                attribValues(attribValIter);
            %use majority voting to assign class
            if class1Count > class2Count
                attribValueClass(attribValIter, 2) = 1;
            else
                attribValueClass(attribValIter, 2) = 2;
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
        
        root.child(attribValIter) = learnDtree(filteredData, ...
                                               filteredLabels, ...
                                               currDepth + 1, ...
                                               maxDepth, ...
                                               filteredAttribs, ...
                                               majLabel);
        root.child(attribValIter).attribValue = attribValues(attribValIter);  
    end
    
    
end

