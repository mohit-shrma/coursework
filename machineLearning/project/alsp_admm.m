function []  = alsp_admm(adjFileName,src_vec,dt_vec,drange)
% Main function to solve the ALSP problem using the ADMM technique. This function inturn calls the ADMM solver to find the shortest path
% adj_mat stores our dynamic graph as a Time Expanded Network
% src_vec is a vector containing the sources (note these source are time expanded)
% dt_vec is a vector containing the destinations (each first destination of a particular source in src_vec)
% drange is a vector containing the range of destination for a particular destination is dt_vec
% lambda is the start time interval.

M = csvread('samp_teg.txt');
%M = [1 2 1; 1 3 1; 2 4 1; 3 4 1];
%cur_source = 1;
%cur_dest = 4;

if length(src_vec) ~= length(dt_vec)
	disp('Error in the input. Source and Destination vectors dont match in length')
end

numNodes = max(max(M(:,1)),max(M(:,2)));
numEdges = size(M,1);

%weights vec
weights=zeros(numEdges, 1);

%constrained matrix
constrainedGraph=sparse(numNodes,numEdges);

for edgeInd=1:size(M,1)
    %TODO: remove curr wt check if in matrix, we remove the '-1' vals
    currWt = M(edgeInd, 3);
    fromNode = M(edgeInd, 1); 
    toNode = M(edgeInd, 2); 
    constrainedGraph(fromNode, edgeInd) = 1;
    constrainedGraph(toNode, edgeInd) =-1; 
    weights(edgeInd) = currWt;
end

for iter = 1: length(src_vec)
	cur_source = src_vec(iter);
	cur_dest  = [];
	cur_dest(1) = dt_vec(iter);

	%Preparing the vector of desination nodes 
	for k = 1:drange-1
		cur_dest(k+1) = cur_dest(k) + 1; 	
    end
    
	b = zeros(numNodes,1);
	b(cur_source) = length(cur_dest); 
	b(cur_dest) = -1;
	
	%Calling ADMM module to find the flow
    
    %initialize weights of flow edges
    w = weights;
    
    %accelaerated admm parameter b/w 1 and 1.8
    alpha = 1;

    %penalty parameter > 0
    rho = 1;
    
    
    [z, history] = admmLinProg(w, A, b, rho, alpha)
    
end  

end


