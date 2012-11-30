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

%make all flow >= 0
A = eye(16);
A = A.*-1;
b = zeros(16,1);

%TODO: make nonexisting flow = 0

%perform standard linear programming
[x,fval,exitflag,output] = linprog(c,A,b,Aeq,beq,[],[],[],optimset('Display','iter'))






