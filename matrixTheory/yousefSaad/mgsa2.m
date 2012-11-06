 function [Q,R] = mgsa2 (A)
% [Q,R] = mgsa (A)
% computes the QR factorization of $A$ via
% Modified Gram Schmid 
% 
 [m,n] = size(A); 
 Q = A;   
 for i=1:n
     t =  norm(Q(:,i),2 ) ;
     Q(:,i) = Q(:,i) / t ;
     R(i,i) = t  ;
%%---------- columns i+1:m 
     for j=i+1:n 
       R(i,j)= Q(:,j)'*Q(:,i);	    
       Q(:,j) = Q(:,j)-R(i,j)*Q(:,i); 
     end
 end 
      
