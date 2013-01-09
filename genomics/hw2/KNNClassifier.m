%KNN classifier
function [predictedLabels] = KNNClassifier(trainingData, ...
                                           trainingLabels, ...
                                           dataToPredict, k)
predictedLabels = zeros(size(dataToPredict,1), 1);
%for each data to work on deduce labels
for dataIter = 1:size(dataToPredict,1)
    kNeighbors = searchKNearestNeighbor(trainingData, ...
                                        dataToPredict(dataIter,:), ...
                                        k);
    %get the labels
    kNeighborsLabels = trainingLabels(kNeighbors);
    
    %index of this matrix represent labels  1 and 2
    labelCount = zeros(2,1);
    for labelIter = 1:size(kNeighborsLabels,1)
        temp = kNeighborsLabels(labelIter);
        labelCount(temp) = labelCount(temp) + 1;
    end
    
    %get the max and the index will contain the predicted labels
    %i.e 1 or 2
    [~, i] = max(labelCount);
    predictedLabels(dataIter) = i;    
end



