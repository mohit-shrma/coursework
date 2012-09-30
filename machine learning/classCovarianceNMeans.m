%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: get in class covariance of data
%}

function [classCovariances, classMeans, classSize, classPriors] = classCovarianceNMeans(data, labels, ...
                                              classes)
sizeData = size(data, 1);
numFeatures = size(data, 2);
classCovariances = zeros(size(classes,1), numFeatures, ...
                         numFeatures);
classSize = zeros(size(classes, 1), 1);
classMeans = zeros(size(classes, 1), numFeatures);
classSums = zeros(size(classes, 1), numFeatures);
classPriors = zeros(size(classes, 1), 1);

%go through datasets and find sum of each class data
for iter=1:sizeData
    
    %get the current data label
    dataLabel = labels(iter);
    
    %find the class label or indice matching class
    classInd = -1;
    for classIter=1:size(classes,1)
        if dataLabel == classes(classIter)
            classInd = classIter;
            break;
        end
    end
    
    %add it to identified class sums
    classSums(classInd, :) = classSums(classInd, :) + data(iter, : ...
                                                      );
    %increment the corresponding class size
    classSize(classInd) = classSize(classInd) + 1;
end

%mean vectors from the sum of data vector
for iter=1:size(classes)
    classMeans(iter, :) = classSums(iter, :)/classSize(iter);
    classPriors(iter) = classSize(iter)/sizeData;
end

for iter=1:sizeData
    
    %get the current data label
    dataLabel = labels(iter);
    
    %find the class label or indice matching class
    classInd = -1;
    for classIter=1:size(classes,1)
        if dataLabel == classes(classIter)
            classInd = classIter;
            break;
        end
    end
    
    varianceVec = (data(iter, :) - classMeans(classInd, :))';
    currVariance = varianceVec * varianceVec';
    classCovariances(classInd, :,:) = classCovariances(classInd, :, :) ...
                      + reshape(currVariance, 1, numFeatures, numFeatures);
end

for iter=1:size(classes,1)
    classCovariances(iter,:,:) = classCovariances(iter,:,:) / classSize(iter);
end
