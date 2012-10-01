%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: compute sigmoid of input
%}

function [Z] = sigmoid(X)
Z = 1.0 ./ (1.0 + exp(-X));