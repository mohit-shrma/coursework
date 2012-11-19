%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 10/25/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: learn decision stumps on training data and apply k-fold ...
       validation where k is passed as an argument to function
%}

function [] = myDstump(dataFileName, numFolds)

load(dataFileName);

myDtree(dataFileName, 1,numFolds);