%load all required data
load all_ppidata.mat

%find degree distribution
[degreeDist, maxDegree, maxI, nodeDegrees] = degreeDistribution(ppi_network);

%plot degree distribution
plot(degreeDist);
xlabel('proteins');
ylabel('degree');

%get the maximum degree protein
maxDegreeProt = genenames(maxI);
fprintf('maximum degree protein is %s, with degree = %d\n', maxDegreeProt{1}, maxDegree);

%clustering coefficients of protein in nodes
clusterCoeffs = clusteringCoeff(ppi_network, nodeDegrees);

%plot clustering coefficient vs degree distribution
plot(clusterCoeffs, nodeDegrees, 'o');
xlabel('clustering coefficient');
ylabel('degree');

%investigate the properties of two proteins, YNL110C (NOP15 ) -> 379 and
%YML085C (TUB1) -> 939
prot1Indx = 379;
prot2Indx = 939;

prot1Name = genenames(prot1Indx);
prot2Name = genenames(prot2Indx);

prot1Degree =nodeDegrees(prot1Indx);
prot2Degree =nodeDegrees(prot2Indx);

prot1ClusterCoeff = clusterCoeffs(prot1Indx);
prot2ClusterCoeff = clusterCoeffs(prot2Indx);

fprintf('protein: %s degree = %d clusterCoeff = %f \n', ...
        prot1Name{1}, prot1Degree, prot1ClusterCoeff);
fprintf('protein: %s degree = %d clusterCoeff = %f \n', ...
        prot2Name{1}, prot2Degree, prot2ClusterCoeff);

%relationship between gene essentiality and the protein-protein
%interaction network, apply ranksum test b/w degrees of essentail
%and non-essential proteins
%p = 6.9841e-26, h = 1 (1 mean to reject the null hypthesis that
%from identical distribution with equal medians)
essentialIndices = find(is_essential == 1);
nonEssentialIndices = find(is_essential == 0);
[p, h] = ranksum(nodeDegrees(essentialIndices), ...
                 nodeDegrees(nonEssentialIndices));


