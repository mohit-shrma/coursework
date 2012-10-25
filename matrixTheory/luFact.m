function [L, U] = luFact(A)
%solves lu factorization
n = size(A, 1);
L = eye(n);
for k=1:n
    L(k+1:n, k) = A(k+1:n, k)/A(k, k);
    for i=k+1:n
       A(i, :) =  A(i, :) - L(i, k)*A(k, :);
    end
end
U = A

