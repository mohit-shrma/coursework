
%load data from file
import_wang_data;

%count the numbers
 
%number of probes in dataset
numProbes = size(Data,1);
fprintf('\nnumber of probes in dataset = %d\n', numProbes);

%number of patient samples in dataset
numPatientSamples = size(Data,2);
fprintf('\nnumber of patients in dataset = %d\n', numPatientSamples);

map_probes_to_genes;

%unique genes in sample
uniqCount=length(unique(GeneID));
fprintf('\nnumber of unique genes = %d\n', uniqCount);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%boxplot before normalization
boxplot(Data(:, 1:4));

%normalize data
normData = quantilenorm(Data);

%boxplot after normalization
boxplot(normData(:, 1:4));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%import java libraries
import java.util.*;
m = HashMap;

%get dimensions of clinical data
clinical_data_siz = size(clinical_data);

%{to create a map for data
%only containing gene name and relapse status
%}
for i=2:size(clinical_data,1)
    m.put(clinical_data{i,2}, clinical_data{i,5});
end

%initialize empty indices to columns with relapse flags
relapseList = [];

%initialize empty indices to columns with non relapse flags
nonrelapseList = [];

%loop through patient data to get the columns in relapse list
for iter=1:size(PatientID,1)
    if m.get(PatientID{iter}) == '1'
	   relapseList = [relapseList, iter];
      else
	nonrelapseList = [nonrelapseList, iter];
    end
end

numRelapses = length(relapseList);
numNonRelapses = length(nonrelapseList);

geneRelapseSamp = [];
geneNonRelapseSamp = [];

probeTTest = [];
probeRankTest = [];

for probeIter=1:size(normData, 1)
    
    geneRelapseSamp = normData(probeIter, relapseList);
    geneNonRelapseSamp = normData(probeIter, nonrelapseList);
    
    %perform ttest2
    [h, p] = ttest2(geneNonRelapseSamp, geneRelapseSamp);
    probeTTest = [probeTTest; [h, p]];
    
    %perform ranksumtest
    [p, h] = ranksum(geneNonRelapseSamp, geneRelapseSamp);
    probeRankTest = [probeRankTest; [h, p]];

end

%sort ttest results based on p-values
[sortedTTest, sortedTTestIndx] = sortrows(probeTTest, 2);

fprintf('\nTop 10 genes selected by ttest along with p values:\n');

%top 10 genes selected by ttest along with pvalues
for iter=1:10
    tempGeneName = GeneID(sortedTTestIndx(iter));
    fprintf('%s\t%d\n', tempGeneName{1}, sortedTTest(iter,2));
end


%sort ranksum results based on p-values
[sortedRankSum, sortedRankSumIndx] = sortrows(probeRankTest, 2);

fprintf('\nTop 10 genes selected by ranksum test along with p values:\n');

%top 10 genes selected by rank test along with pvalues
for iter=1:10
    tempGeneName = GeneID(sortedRankSumIndx(iter));
    fprintf('%s\t%d\n', tempGeneName{1}, sortedRankSum(iter,2));
end

%selected genes with ttest h=1, 2959
fprintf('num of selected genes by ttest = %d\n', length(nonzeros(probeTTest(:, 1))));

%selected genes with ranksum test h=1, 3149
fprintf('num of selected genes by ranktest = %d\n', length(nonzeros(probeRankTest(:, 1))));

%histogram of all pvalues for ttest
hist(probeTTest(:,2));
ylabel('ttest p values');
xlabel('probes');

%histogram of all pvalues for ranksum
hist(probeRankTest(:,2));
ylabel('ranksum p values');
xlabel('probes');

%elements selected in ttest
selectedByTTest = find(probeTTest(:,1));

%elements selected in ranktest
selectedByRankTest = find(probeRankTest(:,1));

%overlapped indices in both test
overlappedIndices = intersect(selectedByTTest, selectedByRankTest);

overlappedUniqueGenes = unique(GeneID(overlappedIndices));
fprintf('number of overlapped non-unique genes in both tests = %d\n', length(overlappedIndices));
fprintf('number of overlapped unique genes in bith tests = %d\n', length(overlappedUniqueGenes));

%fprintf('\noverlapped genes between two differnt approaches as follow:');
%overlappedUniqueGenes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Bonferroni global significance level alphaA=0.05
alphaA = 0.05;
numStatisticalTests = size(normData, 1);

%significance level for single test, alphaS =
% alphaA/numStatisticalTests
alphaS = alphaA/numStatisticalTests;

% perform t-test with above value as significance level for single
% test as per bon ferroni

geneRelapseSamp = [];
geneNonRelapseSamp = [];
tTestForBFerroni = [];

for probeIter=1:size(normData, 1)
    
    geneRelapseSamp = normData(probeIter, relapseList);
    geneNonRelapseSamp = normData(probeIter, nonrelapseList);
    
    %perform ttest2
    [h, p] = ttest2(geneNonRelapseSamp, geneRelapseSamp, alphaS);
    tTestForBFerroni = [tTestForBFerroni; [h, p]];
    
end

%sort ttest results based on p-values
[sortedTTestForBF, sortedTTestForBFIndx] = sortrows(tTestForBFerroni, 2);

%selected genes with ttest h=1, 2
length(nonzeros(tTestForBFerroni(:, 1)));

%histogram of all pvalues for ttest
hist(tTestForBFerroni(:,2));

%elements selected in ttest
selectedByBFTTest = find(tTestForBFerroni(:,1));

fprintf('number of selected genes by bonferroni test = %d\n', ...
        length(selectedByBFTTest));
GeneID(selectedByBFTTest);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%for benjamini and hochberg FDR correction
%global significance level alphaGlobalBH = 0.05
alphaGlobalBH=0.05;
numTests=size(sortedTTest,1);

%for independent genes
bhIndependentGenes = [];

for iter=1:numTests
    pVal = sortedTTest(iter,2);
    if pVal  < (iter*alphaGlobalBH)/numTests
        %report the gene
        bhIndependentGenes = [bhIndependentGenes; sortedTTestIndx(iter) ...
                           pVal];
    end
end

bhIndependentSelectedGenes = GeneID(sortedTTestIndx(1:size(bhIndependentGenes,1)));

adjustedPValIndependent = [1:numTests].*(alphaGlobalBH/numTests);

fprintf('\nnumber of selected genes by BH procedure(independent case) = %d\n',...
    length(bhIndependentGenes));

%for dependent genes
%compare Pi with i*alpha/N*C(N)
bhDependentGenes = [];
inverseSum = 0;

for iter=1:numTests
    inverseSum = inverseSum + (1/iter);
end

%store the denominator constant
denom = numTests*inverseSum;

for iter=1:numTests
    pVal = sortedTTest(iter,2);
    if pVal  < (iter*alphaGlobalBH)/(denom)
        %report the gene
        bhDependentGenes = [bhDependentGenes; sortedTTestIndx(iter) ...
                           pVal];
    end
end

adjustedPValDependent = [1:numTests].*(alphaGlobalBH/denom);

fprintf('\nnumber of selected genes by BH procedure(dependent case) = %d\n', length(bhDependentGenes));

%plot p-vals and adjusted p-vals for independent case
plot(1:numTests, adjustedPValIndependent, 1:numTests, sortedTTest(:,2));
xlabel('sorted probe indices');
ylabel('p-values');
legend('adjusted p-value threshold', 'p value')
title('BH independent');

%plot p-vals and adjusted p-vals for dependent/general case
plot(1:numTests, adjustedPValDependent, 1:numTests, sortedTTest(:,2));
xlabel('sorted probe indices');
ylabel('p-values');
legend('adjusted p-value threshold', 'p value');
title('BH dependent');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%extra credit work by comparing with second study

fprintf('\n******** extra credit work by comparing with second study **********\n');

load('ExtraCredit_vantveer.mat');
norm_vantveer_data = quantilenorm(vantveer_data);
vantveerNonRelapseCols=find(vantveer_labels==-1);
vantveerRelapseCols=find(vantveer_labels==1);
tTestForVantveer = [];


for probeIter=1:size(norm_vantveer_data, 1)
    
    geneRelapseSamp = norm_vantveer_data(probeIter, vantveerRelapseCols);
    geneNonRelapseSamp = norm_vantveer_data(probeIter, vantveerNonRelapseCols);
    
    %perform ttest2
    [h, p] = ttest2(geneNonRelapseSamp, geneRelapseSamp);
    tTestForVantveer = [tTestForVantveer; [h, p]];
    
end


%sort ttest results based on p-values
[sortedTTestForVant, sortedTTestForVantIndx] = sortrows(tTestForVantveer, 2);

%selected genes with ttest h=1, 2
length(nonzeros(tTestForVantveer(:, 1)));

%for benjamini and hochberg FDR correction
%global significance level alphaGlobalBH = 0.05
alphaGlobalBH=0.05;
numVantTests=size(sortedTTestForVant, 1);

%for independent genes
bhVantIndependentGenes = [];

for iter=1:numVantTests
    pVal = sortedTTestForVant(iter,2);
    if pVal  < (iter*alphaGlobalBH)/numVantTests
        %report the gene
        bhVantIndependentGenes = [bhVantIndependentGenes; sortedTTestForVant(iter) ...
                           pVal];
    end
end

%adjustedPValIndependent = [1:numVantTests].*(alphaGlobalBH/numVantTests);

fprintf('\nnumber of selected genes by BH procedure(independent case) in second study = %d\n',...
    length(bhVantIndependentGenes));


%for dependent genes
%compare Pi with i*alpha/N*C(N)
bhVanDependentGenes = [];
inverseSum = 0;

for iter=1:numVantTests
    inverseSum = inverseSum + (1/iter);
end

%store the denominator constant
denom = numVantTests*inverseSum;

for iter=1:numVantTests
    pVal = sortedTTestForVant(iter,2);
    if pVal  < (iter*alphaGlobalBH)/(denom)
        %report the gene
        bhVanDependentGenes = [bhVanDependentGenes; sortedTTestForVant(iter) ...
                           pVal];
    end
end

%adjustedPValDependent = [1:numVantTests].*(alphaGlobalBH/denom);

fprintf('\nnumber of selected genes by BH procedure(dependent case) in second study = %d\n'...
    , length(bhVanDependentGenes));


vantSelectedGenes = vantveer_genes(sortedTTestForVantIndx(1:size(bhVantIndependentGenes,1)));

selectedByVantTTest = find(tTestForVantveer(:,1));

%selectedFromSortedVant = sortedTTestForVantIndx(1:length(find(sortedTTestForVant(:,1))));
%vantSelectedGenes = vantveer_genes(selectedFromSortedVant, 2);

%all genes differentially expressed in both study
%commonGenes = intersect(GeneID(selectedByTTest), vantveer_genes(selectedFromSortedVant, 2));
commonGenes =intersect(GeneID(selectedByTTest), vantveer_genes(selectedByVantTTest, 2));
fprintf('Total number of overlapped genes in both study = %d\n', length(commonGenes));

%select top 100genes
top100VantTTestSelectedGenes = vantveer_genes(sortedTTestForVantIndx(1:100), 2);
top100OldTTestSelectedGenes = GeneID(sortedTTestIndx(1:100));
overlappedGenesInTop100 = intersect(top100VantTTestSelectedGenes, top100OldTTestSelectedGenes);

%fprintf('\nTop 100 selected genes from second study is as follow:\n');
top100VantTTestSelectedGenes;

%NEK2 found which is shown as cancer expressed genes
fprintf('number of overlapped genes in top 100 of both study = %d\n',...
    length(overlappedGenesInTop100));

overlappedGenesInTop100