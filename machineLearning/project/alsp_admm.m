function []  = alsp_admm(adj_mat,src_vec,dt_vec,drange)
% Main function to solve the ALSP problem using the ADMM technique. This function inturn calls the ADMM solver to find the shortest path
% adj_mat stores our dynamic graph as a Time Expanded Network
% src_vec is a vector containing the sources (note these source are time expanded)
% dt_vec is a vector containing the destinations (each first destination of a particular source in src_vec)
% drange is a vector containing the range of destination for a particular destination is dt_vec
% lambda is the start time interval.


if length(src_vec) ~= length(dt_vec)
	disp('Error in the input. Source and Destination vectors dont match in length')
end


[r,c] = size(adj_mat);
%Preparing the mapping for flow vector. Here each entry in the flow vector corresponds for to edge from the Adj mat.
Edges = [];
ecounter = 0;
Edge_mapper = zeros(r, r);
for i = 1:r
	for j = 1:c
		if adj_mat(i,j) > 0
			ecounter = ecounter + 1;
			Edges = [Edges;[i,j]];
			Edge_mapper(i,j) = ecounter;
		end 
	end
end

for iter = 1: length(src_vec)
	cur_source = src_vec(iter);
	cur_dest  = [];
	cur_dest(1) = dt_vec(iter);

	%Preparing the vector of desination nodes 
	for k = 1:length(drange)
		cur_dest(k+1) = cur_dest(k) + 1; 	
	end

	%Code for creating the A matrix. A is the flow constraint matrix with r rows and ecounter cols
	if r ~= c
		disp('Error in creating of Adj matrix. It is not square');	
	end

	A = zeros(r, ecounter);
	for node = 1:r  
		temp_cons = zeros(1,ecounter);
		out_edges = Edge_mapper(node, Edge_mapper(node,:) ~= 0); 		 
		in_edges =  Edge_mapper(Edge_mapper(:,node) ~= 0, node);
		temp_cons(out_edges) = 1;
		temp_cons(in_edges) = -1;
		A(node, :) = temp_cons;
	end

	b = zeros(r,1);
	b(cur_source) = length(cur_dest); 
	b(cur_dest) = -1;
	
	%Calling ADMM module to find the flow
    
    %initialize weights of flow edges
    w = zeros(ecounter, 1);
    for edgeIter=1:length(Edges)
        fromToPair = Edges(edgeIter, :);
        w(edgeIter) = adj_mat(fromToPair(1), fromToPair(2));
    end
    
    %accelaerated admm parameter b/w 1 and 1.8
    alpha = 1;

    %penalty parameter > 0
    rho = 1;
    
    [z, history] = admmLinProg(w, A, b, rho, alpha)
end  

end





function [z, history] = admmLinProg(c, A, b, rho, alpha)
% linprog  Solve standard form LP via ADMM
%
% [x, history] = linprog(c, A, b, rho, alpha);
% 
% Solves the following problem via ADMM:
% 
%   minimize     c'*x
%   subject to   Ax = b, x >= 0
% 
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and 
% dual residual norms, and the tolerances for the primal and dual residual 
% norms at each iteration.
% 
% rho is the augmented Lagrangian parameter. 
%
% alpha is the over-relaxation parameter (typical values for alpha are 
% between 1.0 and 1.8).
%
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

t_start = tic;

% Global constants and defaults

QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

% Data preprocessing

[m n] = size(A);

% ADMM solver

x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER

    % x-update
    %tmp = [ rho*eye(n), A'; A, zeros(m) ] \ [ rho*(z - u) - c; b ];
    tmp = pinv([ rho*eye(n), A'; A, zeros(m) ]) * [ rho*(z - u) - c; b ];
    
    x = tmp(1:n);

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = pos(x_hat + u);

    u = u + (x_hat - z);

    % diagnostics, reporting, termination checks

    history.objval(k)  = objective(c, x);
    
    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));
    
    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && history.s_norm(k) < history.eps_dual(k))
         break;
    end
end

if ~QUIET
    toc(t_start);
end

end

function obj = objective(c, x)
    obj = c'*x;
end
