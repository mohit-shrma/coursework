   function [A,beta] = hoQR(A)
%% function [A,beta] = hoQR(A)
%% computes the householder QR factorization
[m,n] = size(A);
for k=1:n
    [v, bet] = house1(A(k:m,k));
    beta(k) = bet;
    z  = bet * (v' * A(k:m,k:n));
    A(k:m,k:n) = A(k:m,k:n) - v * z;
    if (k < m)
        A(k+1:m,k) = v(2:m-k+1);
    end
end
