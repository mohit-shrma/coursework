%Authors: Mohit Sharma, Farideh Fazayeli

clear all
load('all_ppidata.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Question 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%
numGenes = size(ppi_network,1);
%find degree of each gene
degree = sum(ppi_network);
h = figure;
hist(degree)
xlabel('degree')
ylabel('frequency')
saveas(h, 'degreeDist', 'jpg')



%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Question 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%
CC = zeros(1, numGenes);   % A vector to store Clustering Coefficient for each gene
index = 1:numGenes;
for gene = 1: numGenes %for all genes
    indNgh = index(ppi_network(gene, :) == 1); %find set of its neighbours
    numNgh = length(indNgh); %Number of neighbours 
    edges = length(nonzeros(ppi_network(indNgh,indNgh))); %number of links between them -- ppi_network(indNgh,indNgh) is a subgraph including only gene neighbours
    CC(gene) = edges/(numNgh*(numNgh-1));  %Compute the clustering coefficent
end



msg = sprintf('Total number of nodes in the network: %d', numGenes);
disp(msg)

msg = sprintf('Total number of edges in the network: %d', length(nonzeros(ppi_network))/2);
disp(msg)


msg = sprintf('Number of proteins with degree 1 %d', length(nonzeros(isnan(CC))));
disp(msg)

[maxDeg, maxInd] = max(degree);
msg = sprintf('The highest degree protein %s with %d interactions', genenames{maxInd}, maxDeg);
disp(msg)


h = figure;
scatter(degree, CC);
xlabel('degree')
ylabel('Clustering Coefficient')

saveas(h, 'ClusCoeffVsDegree', 'jpg')


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Question 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%

ind = find(strcmp('YNL110C', genenames));
msg = sprintf('YNL110C: degree = %d, Clustering Coefficient = %f', degree(ind), CC(ind));
disp(msg)

ind = find(strcmp('YML085C', genenames));
msg = sprintf('YML085C: degree = %d, Clustering Coefficient = %f', degree(ind), CC(ind));
disp(msg)

clear gene indNgh numNgh edges h msg maxInd maxDeg ind



%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Question 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%find the difference between essential and non-essential genes in terms of
%degree
EssInd = index(is_essential);

nonEssInd = setdiff(index, EssInd);


freq = [];
EssDeg = degree(EssInd);
freq(:, 1) = histc(EssDeg, min(EssDeg):5:max(EssDeg));
nonEssDeg = degree(nonEssInd);
freq(:, 2) = histc(nonEssDeg, min(EssDeg):5:max(EssDeg));

[p, h, stats] = ranksum(EssDeg, nonEssDeg)
h = figure;
bar(freq)
legend('Essential Genes', 'Non-essential Genes');
xlabel('degree')
ylabel('frequency')

count = 1;
for ii = min(EssDeg):5:max(EssDeg)
    tic{count} = int2str(ii);
    count = count + 1;
end

l = (max(EssDeg)-min(EssDeg))/5;
set(gca,'XTick',1:l+1)
set(gca,'XTickLabel',tic)


saveas(h, 'essDegree', 'jpg');