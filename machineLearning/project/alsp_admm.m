function []  = alsp_admm(adjFileName,src_vec,dt_vec,drange)
% Main function to solve the ALSP problem using the ADMM technique. This function inturn calls the ADMM solver to find the shortest path
% adj_mat stores our dynamic graph as a Time Expanded Network
% src_vec is a vector containing the sources (note these source are time expanded)
% dt_vec is a vector containing the destinations (each first destination of a particular source in src_vec)
% drange is a vector containing the range of destination for a particular destination is dt_vec
% lambda is the start time interval.

M = csvread(adjFileName);
adj_mat = spconvert(M);
if length(src_vec) ~= length(dt_vec)
	disp('Error in the input. Source and Destination vectors dont match in length')
end



[r,c] = size(adj_mat);
%Preparing the mapping for flow vector. Here each entry in the flow vector corresponds for to edge from the Adj mat.
Edges = [];
ecounter = 0;
Edge_mapper = sparse(r, r);
[rowInd, colInd, val] = find(adj_mat);

for iter = 1:length(val)
    if val(iter) > 0
        ecounter = ecounter + 1;
        Edges = [Edges;[rowInd(iter),colInd(iter)]];
        Edge_mapper(rowInd(iter), colInd(iter)) = ecounter;
    end
end

for iter = 1: length(src_vec)
	cur_source = src_vec(iter);
	cur_dest  = [];
	cur_dest(1) = dt_vec(iter);

	%Preparing the vector of desination nodes 
	for k = 1:drange-1
		cur_dest(k+1) = cur_dest(k) + 1; 	
	end

	%Code for creating the A matrix. A is the flow constraint matrix with r rows and ecounter cols
	if r ~= c
		disp('Error in creating of Adj matrix. It is not square');	
	end

	A = sparse(r, ecounter);
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


