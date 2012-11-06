 function [A] = testHW4(n,m,tau)
%% generates a rectangular test matrix. 
%% when tau is close to zero the matrix is 
%% ill-conditioned.
u = [1:n]';   v = [1:m]';    v = 1 ./ v;
A = u * v';
for i=1:min(n,m) 
    A(i,i) = A(i,i) + tau;
end
