function [L, D]  = ldlt (A)
% function [L, U]  =  ldlt (A)
% LDLT factorization of A .
%--------------------------------------------
 n = size(A,1) ;
 for k=1:n-1 
    for i=k+1:n
      piv = A(k,i) / A(k,k) ;
      A(i,i:n)=A(i,i:n)-piv*A(k,i:n);
    end
 end 
 D =diag(A);
 U = diag(D) \ triu(A);
 L = U';

