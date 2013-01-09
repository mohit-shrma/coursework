%find clustering coefficient for each node in network
function [clusteringCoeffs] = clusteringCoeff(network, ...
                                                  nodeDegrees)

%number of proteins
numProteins = size(network, 1);

%clustering coefficients
clusteringCoeffs = zeros(numProteins, 1);

for proteinIter = 1:numProteins
    edges = zeros(nodeDegrees(proteinIter), 1);
    edgeCounter = 0;
    for colIter=1:numProteins
        if network(proteinIter, colIter) == 1
            edgeCounter = edgeCounter + 1;
            edges(edgeCounter, 1) = colIter;
        end
    end
    if size(edges,1) ~= (edgeCounter)
        %TODO: something wrong
        fprintf('edgecount dont corresponds to node degrees\n');
    end
    
    %compute actual number of edges among neighbors
    edgeAmongNeigbors = 0;
    for iterI = 1:size(edges,1)
        for iterJ = iterI+1:size(edges,1)
            if network(edges(iterI), edges(iterJ)) == 1
                edgeAmongNeigbors = edgeAmongNeigbors + 1;
            end
        end
    end
    
    clusteringCoeffs(proteinIter) = 2*edgeAmongNeigbors/ ...
        (edgeCounter*(edgeCounter - 1));
    
end