%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 10/25/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: predict from decision tree by parsing it
%}

function [classLabel] = predictFrmDtree(dataVec, root)

classLabel = -99

if root.numChild == 0
    %encountered leaf return class label
    classLabel = root.classLabel
else
    %check for current node's attribute values
    dataTestAttribValue = dataVec(root.attrib);
    %search for child node having current attribute value
    for childIter=1:root.numChild
        if root.child(childIter).attribValue == dataTestAttribValue
            classLabel = predictFrmDtree(dataVec, root.child(childIter));
    end    
    
    end
if classLabel == -99
    fprintf('some err')
end