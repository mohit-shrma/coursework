%returns weighted sample by resampling proportional to weights
function [weightedSamples] = weightedSampleWithReplacement(samples, ...
                                                  samplesWeight)
%{
   CSci5512 Spring'12 Homework 3
   login: sharm163@umn.edu
   date: 4/11/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: weightedSampleWithReplacement
%}
    
numSamples = size(samples, 1);

%get the samples with false value
falseSamplesInd = (find(samples == 1));

%get the samples with true value
trueSamplesInd = (find(samples == 2));

%get the  weight of both sample
weightFalseSamples = sum(samplesWeight(falseSamplesInd));
weightTrueSamples = sum(samplesWeight(trueSamplesInd));
 
%normalize the samples weight
netWeight = weightFalseSamples + weightTrueSamples;
normalizedFalseWeight = weightFalseSamples/netWeight;
%normalizedTrueWeight = weightTrueSamples/netWeight;

weightedSamples = zeros(numSamples, 1);

%resample using weights
for iter=1:numSamples
    randP = unifrnd(0, 1);
    if randP <= normalizedFalseWeight
        %assign false value denoted by '1'
        weightedSamples(iter) = 1;
    else    
        %assign true value denoted by '2'        
        weightedSamples(iter) = 2;
    end
end
