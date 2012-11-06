 function [Q,R] = cgsa (A)
% [Q,R] = cgsa (A)
% classical Gram Schmidt QR factorization of A
 [m,n] = size(A);
 for j=1:n
     q = A(:,j); 
     a = q;
     for i=1:j-1 
	 r = a'*Q(:,i);	    
         q = q - r*Q(:,i); 
         R(i,j) = r;
     end
%%---------- error exit for case rjj == 0
     r = norm(q) ;
    if (r==0.0), error(' ** zero column'), end
     Q(:,j) = q / r; 
     R(j,j) = r; 
 end 
