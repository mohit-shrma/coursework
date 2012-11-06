function [U]  = cholR (A)
% function [ U ] = cholR (A)
% row outer product cholesky
%---------------------------
 n = size(A,1) ;
 for k=1:n
     if (A(k,k)==0), error(' zero pivot'), end
     A(k,k:n) = A(k,k:n) / sqrt(A(k,k)) ; 
     for i=k+1:n
         A(i,i:n)=A(i,i:n)-A(k,i)*A(k,i:n);
     end
 end 
 U = triu(A); 
