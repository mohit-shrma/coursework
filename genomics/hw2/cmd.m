%script
training_data = importdata('training_data.txt');
training_labels = importdata('training_labels.txt');
test_data = importdata('test_data.txt');
test_labels = importdata('test_labels.txt');

%following is the corrected file for mac os x, it contained '\r', changed
%it to '\n'; tr '\r' '\n' < Gene_Name_24481.txt >
%Gene_Name_24481_corrected.txt
%gene_names = importdata('Gene_Name_24481_corrected.txt');
gene_names = importdata('Gene_Name_24481.txt');

%training data flipped to get 'genes' X 'samples'
training_data_flipped = training_data';


%******************* Q1 ***********************%
%implement the k-means clustering algo
%kMeansClustering implemented as a subroutine in kMeansCluster.m

%***********************************************%

%****************** Q2(a) **********************%
%apply kMeans to cluster genes with k  = 10
[dataCentroidsIndx, centroidsCoord] = kMeansCluster(training_data_flipped,...
                                                    10, 25);
                                                                
clusterSize = countAndPlotClusterSize(dataCentroidsIndx, centroidsCoord);

%apply kMeans to cluster genes with k  = 100

[dataCentroidsIndx, centroidsCoord] = kMeansCluster(training_data_flipped,...
                                                    100, 25);
                                                                
clusterSize = countAndPlotClusterSize(dataCentroidsIndx, centroidsCoord);

%apply kMeans to cluster genes with k  = 200

[dataCentroidsIndx, centroidsCoord] = kMeansCluster(training_data_flipped,...
                                                    200, 25);
                                                                
clusterSize = countAndPlotClusterSize(dataCentroidsIndx, centroidsCoord);


%********************** Q 2(b) *************************%



nonCancerIndices = find(training_labels == 1);
cancerIndices = find(training_labels == 2);

tTestGenes = zeros(size(training_data_flipped), 2);

%traverse each probe and perform t-test on it
for probeIter = 1:size(training_data_flipped, 1)
    geneCancerSamples = training_data_flipped(probeIter, cancerIndices);
    geneNonCancerSamples = training_data_flipped(probeIter,...
                                                    nonCancerIndices);
    %perform ttest2
    [h, p] = ttest2(geneCancerSamples, geneNonCancerSamples);
    tTestGenes(probeIter,:) = [h, p];
end

%sort t-test results based on p-values
[sortedTTestGenes, sortedTTestGenesIndx] = sortrows(tTestGenes, 2);

%select top 1000 genes
selectedGeneIndx = sortedTTestGenesIndx(1:1000);

%select data corresponding to above genes
selectedData = training_data_flipped(selectedGeneIndx, :);

%apply k-means with k=10
[dataCentroidsIndx, centroidsCoord] = kMeansCluster(selectedData, 10,...
                                                                    100);

%count & plot the cluster sizes
clusterSize = countAndPlotClusterSize(dataCentroidsIndx, centroidsCoord);

%apply k-means with k=10
[dataCentroidsIndx, centroidsCoord] = kMeansCluster(selectedData, 20,...
                                                                    100);

%count & plot the cluster sizes
clusterSize = countAndPlotClusterSize(dataCentroidsIndx, centroidsCoord);


%apply k-means with k=10
[dataCentroidsIndx, centroidsCoord] = kMeansCluster(selectedData, 50,...
                                                                    100);

%count & plot the cluster sizes
clusterSize = countAndPlotClusterSize(dataCentroidsIndx, centroidsCoord);


topGenesTrainingData = training_data(:, selectedGeneIndx);
topGenesTestData = test_data(:, selectedGeneIndx);

%****************** Q3 ****************************%
%KNNClassifier implemented as subroutine in KNNClassifier.m
%corresponding functions used written in searchKNearestNeighbor.m,
%pearsonCorrelationCoeff.m
%**************************************************%

% 
% %***************** Q4(a) *************************************************%
% run KNN classifier for k = 1
predictedLabels = KNNClassifier(training_data, training_labels,...
                                                test_data, 1 );

correctPredictions = size(nonzeros(predictedLabels == test_labels), 1);
accuracy = correctPredictions/size(test_data,1);
fprintf('KNNclassifier with k = 1, accuracy = %f\n', accuracy*100);

% run KNN classifier for k = 3
predictedLabels = KNNClassifier(training_data, training_labels,...
                                                test_data, 3 );

correctPredictions = size(nonzeros(predictedLabels == test_labels), 1);
accuracy = correctPredictions/size(test_data,1);
fprintf('KNNclassifier with k = 3, accuracy = %f\n', accuracy*100);

% run KNN classifier for k = 5
predictedLabels = KNNClassifier(training_data, training_labels,...
                                                test_data, 5 );

correctPredictions = size(nonzeros(predictedLabels == test_labels), 1);
accuracy = correctPredictions/size(test_data,1);
fprintf('KNNclassifier with k = 5, accuracy = %f\n', accuracy*100);


%*************************************************************************%

%******************* Q4(b) ***********************************************%
%run KNNClassifier for various k values
%for k = 1
predictedLabels = KNNClassifier(topGenesTrainingData, training_labels,...
                                                topGenesTestData, 1 );

correctPredictions = size(nonzeros(predictedLabels == test_labels), 1);
accuracy = correctPredictions/size(test_data,1);
fprintf('KNNclassifier with k = 1 for top genes, accuracy = %f\n',...
                                                                accuracy*100);

%for k = 3
predictedLabels = KNNClassifier(topGenesTrainingData, training_labels,...
                                                topGenesTestData, 3);

correctPredictions = size(nonzeros(predictedLabels == test_labels), 1);
accuracy = correctPredictions/size(test_data,1);
fprintf('KNNclassifier with k = 3 for top genes, accuracy = %f\n',...
                                                                accuracy*100);

%for k = 5
predictedLabels = KNNClassifier(topGenesTrainingData, training_labels,...
                                                topGenesTestData, 5);

correctPredictions = size(nonzeros(predictedLabels == test_labels), 1);
accuracy = correctPredictions/size(test_data,1);
fprintf('KNNclassifier with k = 5 for top genes, accuracy = %f\n',...
                                                                accuracy*100);

%******************************** Q5 *****************************************%
                                                     
%for svm wit full training data
svmStruct = svmtrain(training_data, training_labels, 'kernel_function',...
                                                                'linear');
predictedLabels = svmclassify(svmStruct, test_data);

correctPredictions = size(nonzeros(predictedLabels == test_labels), 1);
accuracy = correctPredictions/size(test_data,1);
fprintf('svm classifier training with full training data, accuracy = %f\n',...
                                                                accuracy*100);                                                          
%train svm with top 1000 genes selected by ttests
svmStruct = svmtrain(topGenesTrainingData, training_labels, 'kernel_function',...
                                                                'linear');
predictedLabels = svmclassify(svmStruct, topGenesTestData);

correctPredictions = size(nonzeros(predictedLabels == test_labels), 1);
accuracy = correctPredictions/size(test_data,1);
fprintf('svm classifier training with top genes, accuracy = %f\n',...
                                                                accuracy*100);

