%function to count cluster sizes and plot a histogram for it
function [clusterSize] = countAndPlotClusterSize(dataCentroidsIndx, ...
                                                 centroidsCoord)
clusterSize = zeros(size(centroidsCoord, 1), 1);
for iter = 1:size(clusterSize, 1)
    clusterSize(iter) = size(nonzeros(find(dataCentroidsIndx == ...
                                           iter)), 1);
end

%plot the histogram for cluster sizes
bar(clusterSize);
xlabel('centroids'), ylabel('size of clusters');
