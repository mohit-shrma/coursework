%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: use gaussian params to predict label for data
       l(x) = sum<i>log(p(x|y=T)/p(x|y=F))  + log(p(y=T)/p(y=F)) = ...
              sumLogRatio + classPriorRatio
%}

function [classLabel] = naiveBayesPredict(classes, dataRow, ...
                                          gaussianParams, classPriors)


numFeatures = size(dataRow, 2);

sumLogRatio = 0;

classPriorRatio =  classPriors(classes(1))/classPriors(classes(2));

%first compute sum log-ratio for each feature
for featureIter=1:numFeatures
    muNum = gaussianParams(classes(1), featureIter, 1);
    sigmaNum = gaussianParams(classes(1), featureIter, 2);
    if sigmaNum == 0
        if dataRow(featureIter) ~= muNum
            RatioNum = 0;
        else
            RatioNum = 1;
        end
    else
        RatioNum = normpdf(dataRow(featureIter), muNum, sigmaNum);
    end
    
    
    muDen = gaussianParams(classes(2), featureIter, 1);
    sigmaDen = gaussianParams(classes(2), featureIter, 2);
    if sigmaDen == 0
        if dataRow(featureIter) ~= muNum
            RatioDen = 0;
        else
            RatioDen = 1;
        end
    else
        RatioDen = normpdf(dataRow(featureIter), muDen, sigmaDen);    
    end
    
    ratio = RatioNum/RatioDen;
    
    sumLogRatio = sumLogRatio + log(ratio);
end

predictedVal = sumLogRatio + log(classPriorRatio);

if predictedVal > 0
    classLabel = classes(1);
else
    classLabel = classes(2);
end

