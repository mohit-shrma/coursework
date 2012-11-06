  function [Q,R] = mgsa (A)
% [Q,R] = mgsa (A)
% Modified Gram Schmidt QR factorization of A
 [m,n] = size(A); 
 for j=1:n
      q = A(:,j);
      for i=1:j-1 
	 r = q'*Q(:,i);	    
         q = q - r*Q(:,i) ;
         R(i,j) =r;  
     end
     r = norm(q) ;
%%---------- error exit for case rjj == 0
     if (r==0.0), error('** zero column'), end
     Q(:,j) = q / r; 
     R(j,j) = r; 
 end 
