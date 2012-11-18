%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 10/25/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: class definition of decision tree node
%}


classdef Node < handle
    properties
        attribute
        splitVal
        predLabel
    end
    properties (SetAccess = private)
        left
        right
        depth
    end
    
    methods
        function node = Node(attribute, splitVal, predLabel)
        % Node constructs a node
            if nargin > 0
                %TODO: what is nargin
                node.attribute = attribute;
                node.splitVal = splitVal;
                node.predLabel = predLabel;
                node.depth = 0;
            end
        end
        
        function addLeftChild(newNode, leftChild)
        %add the new node as left child of parent
            newNode.left = leftChild;
            if newNode.depth < leftChild.depth + 1
                newNode.depth = leftChild.depth + 1;
            end
        end
        
        function addRightChild(newNode, rightChild)
        %add the new node as right child of parent
            newNode.right = rightChild;
            if newNode.depth < rightChild.depth + 1
                newNode.depth = rightChild.depth  + 1;
            end
        end
        
        function [ret] = isLeaf(node)
        %return true if current node is leaf
            if isempty(node.left) && isempty(node.right)
                ret = true;
            else
                ret = false;
            end
        end
        
        function levelOrderTraversal(node)
        %traverse node subtree level by level, BFS
            queue = [node];
            currDepth = node.depth;
            fprintf('\nheight = %d\n', currDepth);
            while ~isempty(queue)
                %pop the node at end of queue
                poppedNode = queue(end);
                queue = queue(1:end-1);
                if poppedNode.depth ~= currDepth
                    fprintf('\nheight = %d\n', poppedNode.depth);
                    currDepth = poppedNode.depth; 
                end
                if ~poppedNode.isLeaf()
                    %print the attribute of current popped node
                    fprintf(' attribute:%d splitValue:%d ',...
                        poppedNode.attribute, poppedNode.splitVal);        
        
                    %add child nodes to queue
                    queue = [poppedNode.left poppedNode.right  queue];
                else
                    %leaf encountered
                    fprintf(' %d', poppedNode.predLabel);
                end
            end
            fprintf('\n');
        end
        
        function [predictedLabel] =  predictLabel(node, dataVec)
        %predict the label from passed node subtree
            predictedLabel = -99;
            if node.isLeaf()
                %encountered leaf return class label
                predictedLabel = node.predLabel;
            else
                %check for current node's attribute values
                dataTestAttribValue = dataVec(node.attribute);
                if dataTestAttribValue < node.splitVal
                    %predict from left subtree of current node
                    predictedLabel = node.left.predictLabel(dataVec);
                else
                    %predict from right subtree of current node
                    predictedLabel = node.right.predictLabel(dataVec);                    
                end
            end
        end
        
        
        
    end %end methods
end %end classdef