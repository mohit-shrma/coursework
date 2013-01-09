%function that compute k clusters on given data and runs for given maximum
%iterations, returns centroids corresponding to each data point and
%centroids coordinates
function [dataCentroidsIndx, centroidsCoord] = kMeansCluster(data, k, maxIter)

numSamples = size(data,1);

%generate k random points for initialization from training data
centroidsCoordInd = unique(randi(numSamples, k, 1));

%make sure all random points are unique
while size(centroidsCoordInd) < k
    requiredRands = unique(randi(numSamples,k ,1));
    centroidsCoordInd = unique([centroidsCoordInd; requiredRands]);
    centroidsCoordInd = centroidsCoordInd(1:k);
end

%array storing centroids
centroidsCoord =  data(centroidsCoordInd, :);

%array to hold centroid coords for each data point
dataCentroidsIndx = zeros(numSamples, 1);

for iter = 1:maxIter
    
    %boolean variable to track changes in coords
    change = false;
    
    %go through each data set and assign it to centroid
    for dataIter = 1:numSamples
        %evaluate each data point for each initialized centroid
    
        %store temp max pearson index
        tempMaxPearsInd = 1;
    
        %store temp Max pears
        tempMaxPears = corr2(data(dataIter, :),...
                                centroidsCoord(tempMaxPearsInd, :));
        for kIter = 2:k
            coeff = corr2(data(dataIter, :), centroidsCoord(kIter, :));
            if coeff > tempMaxPears
                tempMaxPearsInd = kIter;
                tempMaxPears = coeff;
            end
        end
    
        %update centroid assignment for current datapoint
        dataCentroidsIndx(dataIter) = tempMaxPearsInd;
    
    end

    %update the centroid to new average of points data points per
    %centroid
    for kIter = 1:k
        
        indToSum = find(dataCentroidsIndx == kIter);
        
        if size(indToSum,1) == 0
            continue;
        end
        
        sumToAvg = zeros(1, size(data,2));
        
        for i=1:size(indToSum, 1)
            sumToAvg = sumToAvg + data(i, :);
        end
        
        avg = sumToAvg/size(indToSum, 1);
        
        %check if 98% similar to previous centroid then no need to update
        %as it means change is not and it could conversge to some point
        %after some time of iteration
        correlCoeff = corr2(avg, centroidsCoord(kIter, :));
        if correlCoeff < 0.98
            centroidsCoord(kIter, :) = avg;
            change = true;
        end
        
    end
    
    %if no change in centroids coordinates, then break
    if ~change
        break;
    end
    
end



