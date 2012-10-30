%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 10/25/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: print decision tree by applying bfs
%}

function [] = printDtree(root)

queue = [root];

currDepth = 0;
fprintf('\ndepth = %d\n', 0);
while ~isempty(queue)
    %pop the node at end of queue
    node = queue(end);
    queue = queue(1:end-1);
    
    if node.depth ~= currDepth
            fprintf('\ndepth = %d\n', node.depth);
            currDepth = node.depth;
    end
    
    if node.numChild ~= 0   
        
        %print the attribute of current popped node
        fprintf(' attribute:%d', node.attrib);        
        
        %add child nodes to queue
        for childIter=1:node.numChild
            queue = [node.child(childIter) queue];
        end
    else
        %leaf encountered
        fprintf(' %d', node.classLabel);
    end
        
end
