%Authors: Mohit Sharma, Farideh Fazayeli

function kclq = kClique()
%return all cliques of size k in a matrix of k-dimension

load('all_ppidata.mat')

k = 5;

Adj = ppi_network;
%clear ppi_network
numGenes = size(Adj,1);
%find degree of each gene
degree = sum(Adj);

index = 1: numGenes;
Idx = index(degree >= k);

orgIdx = Idx;

oldIdx = index;

%Remove nodes with degree less than k recursively

while (setdiff(oldIdx, Idx))  
    oldIdx = Idx;
    %remove nodex with degree less than k
    Adj = Adj(Idx, Idx);
    degree = sum(Adj);

    Idx = index(degree >= k);
    orgIdx = orgIdx(Idx);
end


%for all remaining nodes, we will run apriori as follow:
%generate candidate k-cliques from merging (k-1)-cliques with all nodes (avoid
%redunduncy by following order)
%Remove those that are not clique
%repeat above procedure to get all of 5-cliques
numGenes = size(Adj,1);
cliques = {};
cliques{1} = 1:numGenes';

for k = 2: 5
    count = 1;
    kclq = [];
    k1clq = cliques{k-1};
    for ii = 1 : length(k1clq)
        if k == 2
            %find the latest index in the k-1 clique to add index after
            %that to generate k-cliques
            last = k1clq(ii);
        else
            last =  k1clq(ii, end);
        end
        
        for jj = last + 1: numGenes
            if k == 2
                ind = k1clq(ii);
            else
                ind =  k1clq(ii, :);
            end
            %ind: candidate k-cliques
            ind(end+1) = jj;
            len = length(ind);
            %find the number of edges
            edges = sum(sum(Adj(ind, ind)));
            %check if it is a clique
            if edges == (len*(len-1))
                kclq(count, :) = ind;
                count = count +1;
            end
        end
    end
    cliques{k} = kclq;
end

k = 5;
kclq = cliques{k};
%find the index in the ppi_network
kclq = orgIdx(kclq);


%return 3 random cliques
tmp = kclq([100, 2000, 20000], :);
for i = 1 : 3
    clqGeneNames = {genenames{tmp(i, :)}};
    fprintf('Name of genes: %s %s %s %s %s \n', clqGeneNames{1}, clqGeneNames{2}, clqGeneNames{3}, clqGeneNames{4}, clqGeneNames{5});
end

end
