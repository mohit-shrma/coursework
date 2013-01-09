%compute pearson correlation coefficient of two vector
function [pearsonCoeff] = pearsonCorrelationCoeff(X , Y)

if (size(X,1) ~= size(Y,1)) && (size(X,2) ~= size(Y,2))
    fprintf('bad parameters to pearson');
    return
end


meanX = mean(X);
meanY = mean(Y);

xMeanDiff = X - meanX;
yMeanDiff = Y - meanY;

numerator = sum(xMeanDiff.*yMeanDiff);
denominator = sqrt(sum(xMeanDiff.^2) * sum(yMeanDiff.^2)); 


%take care of divide by zero
if denominator == 0
    %transform each point by (1,1) this shouldn't effect similarity or 
    %shape of curves
    xMeanDiff = X - meanX + 1;
    yMeanDiff = Y - meanY + 1;

    numerator = sum(xMeanDiff.*yMeanDiff);
    denominator = sqrt(sum(xMeanDiff.^2) * sum(yMeanDiff.^2)); 
end


pearsonCoeff = numerator/denominator;