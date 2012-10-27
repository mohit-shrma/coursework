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

while length(queue) > 0
    %pop the node at end of queue
    node = queue(end);
    queue = queue(1:end-1);
    
    if node.numChild != 0
        %print the attribute of current popped node
        fprintf('\n attribute : %d', node.attrib);
        %add child nodes to queue
        for childIter=1:numChild
            queue = [queue node.child(childIter)];
        end
    else
        %leaf encountered
        fprintf('\n class label: %d', node.classLabel);
    end
        
end
