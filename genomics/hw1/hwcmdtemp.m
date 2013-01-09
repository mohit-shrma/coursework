%Bonferroni global significance level alphaA=0.05
alphaA = 0.05;
numStatisticalTests = size(normData, 1);

%significance level for single test, alphaS =
% alphaA/numStatisticalTests
alphaS = alphaA/numStatisticalTests;

% perform t-test with above value as significance level for single
% test

probeTTest2 = [];

for probeIter=1:size(normData, 1)
    
    geneRelapseSamp = [];
    geneNonRelapseSamp = [];
    
    for iter=1:numRelapses
        %%do computation on normData(probeIter, relapseList(iter))
        geneRelapseSamp = [geneRelapseSamp, normData(probeIter, ...
                                                 relapseList(iter))];
    end
    
    for iter=1:numNonRelapses
        %%do computation on normData(probeiter, nonrelapseList(iter))
        geneNonRelapseSamp = [geneNonRelapseSamp, normData(probeIter, ...
                                                      nonrelapseList(iter))];
    end
    
    %perform ttest2
    [h, p, ci] = ttest2(geneNonRelapseSamp, geneRelapseSamp, alphaS);
    probeTTest2 = [probeTTest2; [h, p, ci]];
    
end

%sort ttest results based on p-values
[sortedTTest2, sortedTTestIndx2] = sortrows(probeTTest2, 2);

%selected genes with ttest h=1, 
selectedGeneCountTest2 = length(nonzeros(probeTTest2(:, 1)));

%histogram of all pvalues for ttest
hist(probeTTest2(:,2));

%elements selected in ttest
selectedByTTest2 = find(probeTTest2(:,1));
