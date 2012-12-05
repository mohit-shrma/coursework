function []  = alsp_admm(adjFileName,src_vec,dt_vec,drange)
% Main function to solve the ALSP problem using the ADMM technique. This function inturn calls the ADMM solver to find the shortest path
% adj_mat stores our dynamic graph as a Time Expanded Network
% src_vec is a vector containing the sources (note these source are time expanded)
% dt_vec is a vector containing the destinations (each first destination of a particular source in src_vec)
% drange is a vector containing the range of destination for a particular destination is dt_vec
% lambda is the start time interval.
tic
sparseAdjaMatrix = csvread(adjFileName);
%TODO: remove below line after -ve remove for incoming edges
sparseAdjaMatrix = sparseAdjaMatrix(sparseAdjaMatrix(:,3) > 0, :);

%M = [1 2 5; 1 3 1; 2 4 1; 3 4 1];
%cur_source = 1;
%cur_dest = 4;

if length(src_vec) ~= length(dt_vec)
	disp('Error in the input. Source and Destination vectors dont match in length')
end

numNodes = max(max(sparseAdjaMatrix(:,1)),max(sparseAdjaMatrix(:,2)));
numEdges = size(sparseAdjaMatrix,1);

%cost vec
cost=zeros(numEdges, 1);

%constraints matrix
A=sparse(numNodes,numEdges);

for edgeInd=1:size(sparseAdjaMatrix,1)
    %TODO: remove curr wt check if in matrix, we remove the '-1' vals
    currWt = sparseAdjaMatrix(edgeInd, 3);
    fromNode = sparseAdjaMatrix(edgeInd, 1); 
    toNode = sparseAdjaMatrix(edgeInd, 2); 
    A(fromNode, edgeInd) = 1;
    A(toNode, edgeInd) =-1; 
    cost(edgeInd) = currWt;
end

fprintf('completed preprocessing.... \n');
toc

for iter = 1: length(src_vec)
	tic;
	cur_source = src_vec(iter);
	cur_dest  = zeros(drange, 1);
	cur_dest(1) = dt_vec(iter);

	%Preparing the vector of desination nodes 
	for k = 1:drange-1
		cur_dest(k+1) = cur_dest(k) + 1; 	
        end
    
	b = zeros(numNodes,1);
	b(cur_source) = 1;%length(cur_dest); 
	b(cur_dest) = -1;
	
	%Calling ADMM module to find the flow
    
    %initialize weights of flow edges
    w = cost;
    
    %accelaerated admm parameter b/w 1 and 1.8
    alpha = 1;

    %penalty parameter > 0
    rho = 1;

    fprintf('before calling... admm\n');
    toc
    
    tic;
    [z, history] = admmLinProg(w, A, b, rho, alpha)
    z ~= 0
    sum(z)
    fprintf('completed an admm iteration....\n');
    toc
end  



