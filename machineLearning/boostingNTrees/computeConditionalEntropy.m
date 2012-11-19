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
fractionOfDataConsidered = (class1Count+class2Count)/totalSize;
currClass1Fraction =  class1Count/currentNetCount;
currClass2Fraction =  class2Count/currentNetCount;
if currClass1Fraction == 0 || currClass2Fraction == 0 || currentNetCount == 0
    condnEntropy = 0;
else
    condnEntropy = fractionOfDataConsidered *((-currClass1Fraction* ...
                                  log2(currClass1Fraction)) ...
                                  + (-currClass2Fraction* ...
                                  log2(currClass2Fraction)));

end


