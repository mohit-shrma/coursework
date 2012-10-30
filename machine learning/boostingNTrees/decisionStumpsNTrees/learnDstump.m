%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 10/25/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: learn decision stump on training data and return stump ...
       attribute with corresponding class-values pair
%}


function [stumpAttrib, stumpAttribValueClass] = learnDstump(data, ...
                                                  labels)
%get the number of attributes
numAttribs = size(data, 2);

%get the size of data
sizeData = size(data, 1);

%store the best attribute to split on
stumpAttrib = -1;
stumpAttribValueClass = [];
bestCondnEntropy = 1000;

%search for the best attribute to split
for attribIter=1:numAttribs
    attribValues = unique(data(:, attribIter));
    attribValueClass = zeros(length(attribValues), 2);
    %compute conditional entropy and class count corresponding to
    %each attribute
    conditionalEntropy = 0;
    for attribValIter=1:size(attribValues, 1)
        %get the indices having current attribute value
        dataInd = find(data(:, attribIter) == ...
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
        stumpAttrib = attribIter;
        stumpAttribValueClass = attribValueClass;
        bestCondnEntropy = conditionalEntropy;
    end
end

