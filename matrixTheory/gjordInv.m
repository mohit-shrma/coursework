function [invA] = gjordInv(A)
    
    % get matrix dimension
    n = size(A, 1);
    
    % U matrix by gauss-jordan elimination
    U = zeros(n, n);
    
    %Diagonalize A and get U
    for i=1:n
        U(:,i) = A(:,i)/A(i,i);
        U(i,i) = 0;
        for k = 1:n
            if k~= i
                A(k, :) = A(k, :) - U(k, i)*A(i,:);
            end
        end
    end
    
    %initialize B to identity
    B = eye(n);
    % V' matrix, lower triangular 
    V = zeros(n,n);
    % compute V' matrix recursively using part(c) method recursively
    for i = 1:n
        V(i, :) = zeros(1,n);
        sum = zeros(1, n);
        for j=1:i-1
            sum = sum + eMat(i,n)' * U(:, j) * V(j, :); 
        end
        V(i, :) = eMat(i,n)' - sum;
    end
    
    % compute M = I-UV'
    M =  eye(n) - U*V;
    % A is already diagonalized assign it to D
    D = A;
    
    %compute inverse of D
    invD = zeros(3);
    for i = 1:n
        invD(i,i) = 1/(D(i,i));
    end
    
    %compute inverse of A
    invA = invD*M;
   
    %return e vector
    function [e] = eMat(ind, n)
        e = zeros(n, 1);
        e(ind) = 1;