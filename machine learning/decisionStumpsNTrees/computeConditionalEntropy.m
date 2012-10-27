%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 10/25/2012
   name: Mohit Sharma
   id: 4465482
   algorithm:compute conditional entropy based on passed counts, lesser the
             condn entropy better the split 
%}


function[condnEntropy] = computeConditionalEntropy(totalSize, ...
                                                  class1Count, ...
                                                  class2Count)
currentNetCount = class1Count + class2Count;
fractionOfDataConsidered = (class1Count+class2Count)/totalSize
condnEntropy = fractionOfDataConsidered *((-class1Count* ...
                                          log2(class1Count/currentNetCount)) ...
                                          + (-class2Count* ...
                                          log2(class2Count/currentNetCount)))

