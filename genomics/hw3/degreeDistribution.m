%find degree distribution for each node in the network
function [degreeDist, maxDegree, maxI, nodeDegrees] = degreeDistribution(network)

   %number of proteins
   numProteins = size(network, 1);
   
   %degree distribution of nodes
   nodeDegrees = zeros(numProteins, 1);

   for rowIter = 1:numProteins
      for colIter = 1:rowIter-1
	 if network(rowIter, colIter) == 1
	    nodeDegrees(rowIter) = nodeDegrees(rowIter) + 1;
            nodeDegrees(colIter) = nodeDegrees(colIter) + 1;
	 end
      end   
   end

   [maxDegree, maxI] = max(nodeDegrees);
   degreeDist = zeros(maxDegree, 1);

   for iter = 1:size(nodeDegrees, 1)
      currDegree = nodeDegrees(iter);
      degreeDist(currDegree) = degreeDist(currDegree) + 1;
   end

   degreeDist = degreeDist / numProteins;


		   

   
   
