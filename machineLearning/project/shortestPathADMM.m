%we want to compute the the flow matrix  X's, satisfying flow
%conservation constraint
%for computation we have vectorized the flow matrix
%build the constraint equation, Aeq * X = beq
Aeq = zeros(4,16);

Aeq(1, 2) = 1;
Aeq(1, 3) = 1;

Aeq(2, 2) = -1;
Aeq(2, 8) = 1;

Aeq(3, 3) = -1;
Aeq(3, 12) = 1;

Aeq(4, 8) = -1;
Aeq(4, 12) = -1;

beq = [1 0 0 -1]';

%initialize weights of flow edges
c = zeros(16, 1);
c(2) = 1;
c(3) = 1;
c(8) = 8;
c(12) = 2;

A = eye(16);
A = A.*-1;
b = zeros(16,1);

alpha = 1;
rho = 0.5;

%run admm using linear programming from 'Stephen Boyd'
[x, history] = admmLinProg(c, Aeq, beq, rho, alpha)







