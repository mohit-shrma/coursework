%{
   CSci5525 Fall'12 Homework 1
   login: sharm163@umn.edu
   date: 9/29/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: compute softmax/(posterior prob) vector  across all classes and return the ...
       result as vector for a single data row
       Ak(x) = (Wk)'x + Wko, 
       Wk = inv(covariance)*mean(k), 
       Wko= -0.5mean(k)'*inv(covar)*mean(k) + ln(prior)
%}

function [posteriorVec, maxLabel] = softMax(dataRowVec, classMeans, ...
                                            sharedCovariance, ...
                                            classPriors)

classSize = size(classMeans, 1);
posteriorVec = zeros(classSize, 1);
AkVec = zeros(classSize, 1);
invCovar = inv(sharedCovariance);

for iter=1:classSize
    %TODO: verify that transpose of mean vector needs to be done
    Wk = invCovar*classMeans(iter,:)';
    Wko = (-0.5*classMeans(iter,:)*invCovar*classMeans(iter,:)') + ...
          log(classPriors(iter));
    AkVec(iter) = Wk'*dataRowVec' + Wko;
    posteriorVec(iter) = exp(AkVec(iter));
end

%normalize the posterior vec
posteriorVec = posteriorVec/sum(posteriorVec);
[~, maxLabel] = max(posteriorVec);
