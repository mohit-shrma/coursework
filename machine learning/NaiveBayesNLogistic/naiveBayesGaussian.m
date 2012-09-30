%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: apply naive bayes with multivariate gaussian and use ...
       100 random 80-20 train test split to evaluate
%}



%load given data
%TODO: read from commandline, the data file and then load it
dataFileName = 'pima.mat';
load(dataFileName);

%get size of dataset
sizeData = size(data,1);




