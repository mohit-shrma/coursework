  function x = backsolv(A,b) 
% function x = backsolv(A,b) 
% Solves an upper triangular system
% by back-substitution. 
%----------------------------------
 n = size(A,1); 
 x = zeros(n,1);
 for i=n:-1:1
  x(i)=(b(i)-A(i,i+1:n)* x(i+1:n)) / A(i,i);
 end 

