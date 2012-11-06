  function [x]  = gauss (A, b)
% function [x]  = gauss (A, b)
% solves A x  = b by Gaussian elimination
%------------------------------------------
 n = size(A,1) ;
 A = [A,b] 
 pause
 for k=1:n-1
    fprintf(1,'\n \n step %3d \n',k) 
    for i=k+1:n
      piv = A(i,k) / A(k,k) ;
%%A(i,k+1:n+1)=A(i,k+1:n+1)-piv*A(k,k+1:n+1);
      A(i,k:n+1)=A(i,k:n+1)-piv*A(k,k:n+1);
    end
 A 
 pause
 end 
 x = backsolv(A,A(:,n+1)); 
 
