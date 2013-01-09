%function that search for k-nearest neighbors from a given data and
%query point, it returns the indexes with in the data which satisfy
%k-nearest neighbors

function [kNeighbors] = searchKNearestNeighbor(data, query, k)

%if query vector has same dimensions as each row vector in data
if size(data,2) == size(query,2) && k <= size(data,1)
    
    pearsonVal = zeros(size(data,1), 1);
    
    for iter = 1 : size(data,1)
        pearsonVal(iter) = pearsonCorrelationCoeff(data(iter,:), ...
                                                      query);
    end
    
    %sort the pearson vals from current query giving prefernce to
    %larger as it indicates stronger similarity
    [~, orderedIndex] = sort(pearsonVal, 'descend');
   
    %get the k neighbors with shortest distance from current point
    kNeighbors = orderedIndex(1:k);    
else
    kNeighbors = 0;
end



