 function [L] = cholC (A) 
%---------------------------------------------
% function [A] = cholC(A) 
% performs column cholesky factorization of A
% returns a low. triang. matrix.
%---------------------------------------------
 n = size(A,1) ;
 for k=1:n
     A(k,k) = sqrt(A(k,k)) ;
     A(k+1:n,k) = A(k+1:n,k)/A(k,k);
     for i=k+1:n
         A(i:n,i)=A(i:n,i)-A(i,k)*A(i:n,k);
     end
 end
 L = tril(A)
