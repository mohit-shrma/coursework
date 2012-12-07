function []  = alsp_admm(adjFileName,src_vec,dt_vec,drange)
% Main function to solve the ALSP problem using the ADMM technique. This function inturn calls the ADMM solver to find the shortest path
% adj_mat stores our dynamic graph as a Time Expanded Network
% src_vec is a vector containing the sources (note these source are time expanded)
% dt_vec is a vector containing the destinations (each first destination of a particular source in src_vec)
% drange is a vector containing the range of destination for a particular destination is dt_vec
% e.g usage: alsp_admm('asd', [1],[50],[1])

%open the file to wrte the output
fid = fopen(strcat(adjFileName, '.out'), 'w');

tic;

%load spTrustNw.mat
%read the adjacency matrix
sparseAdjaMatrix = csvread(adjFileName); %data_dump
fprintf(fid, 'Adjacency matrix reading finished...\n');
recordElapsedTime(fid);
                             
% sparseAdjaMatrix = [1 2 1; 1 3 3; 2 4 4; 3 4 1; 2 3 1];
% src_vec = [1];
% dt_vec = [4];

if length(src_vec) ~= length(dt_vec)
    fprintf(fid, 'Error in the input. Source and Destination vectors dont match in length');
end

tic

%get the number of nodes
numNodes = max(max(sparseAdjaMatrix(:,1)),max(sparseAdjaMatrix(:, 2)));

%get the number of edges
numEdges = size(sparseAdjaMatrix,1);

%cost vec
cost=zeros(numEdges, 1);

%constraints matrix initialize
A=sparse(numNodes,numEdges);

%build constraints matrix
for edgeInd=1:size(sparseAdjaMatrix,1)
    currWt = sparseAdjaMatrix(edgeInd, 3);
    fromNode = sparseAdjaMatrix(edgeInd, 1); 
    toNode = sparseAdjaMatrix(edgeInd, 2); 
    A(fromNode, edgeInd) = 1;
    A(toNode, edgeInd) =-1; 
    cost(edgeInd) = currWt;
end

fprintf(fid,'completed preprocessing.... \n');
recordElapsedTime(fid);

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

    fprintf(fid,'before calling... admm\n');
    recordElapsedTime(fid);
    
    tic;
    [z, history] = admmLinProg(w, A, b, rho, alpha);
    fprintf(fid,'completed an admm iteration....\n');
    recordElapsedTime(fid);
    
    %get number of iterations and final objective function value
    numIterations = length(history.objval);
    fprintf(fid, 'num of iterations: %d\n', numIterations);
    fprintf(fid, 'final objective func val: %f\n',...
           history.objval(numIterations));
    
    %save non-zero flows in a file, saving flow greater than 10% 
    computedEdgeFlows = [sparseAdjaMatrix(z>=0.1, :) z(z>0.1)];
    
    %save flow matrix to a file
    dlmwrite(strcat('edgeFlows_', adjFileName), computedEdgeFlows, ...
             'delimiter', '\t')
    
end  
fclose(fid);


%record time since last tic into passes output file
function [] =  recordElapsedTime(fid)
fprintf(fid, 'Elapsed time is %f seconds,\n', toc);
