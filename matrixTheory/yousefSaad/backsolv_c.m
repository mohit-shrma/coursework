  function x = backsolv_c(A,b) 
%----------------------------------
% function x = backsolv(A,b) - COLUMN
% Solves an upper triangular system
% by back-substitution. 
%----------------------------------
 n = size(A,1); 
 x = b;
  for i=n:-1:1
     x(i) = x(i)/A(i,i);
     x(1:i-1) = x(1:i-1)-x(i)*A(1:i-1,i);
 end 
%%

