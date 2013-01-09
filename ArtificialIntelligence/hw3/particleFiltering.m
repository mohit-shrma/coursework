%returns new samples for next time step
function [newSamples] = particleFiltering(currEvidence, oldSamples, ...
                                         transitionR, evidenceU)

numSamples = size(oldSamples, 1);
weightSamples = zeros(numSamples, 1);
newSamples = zeros(numSamples, 1);

for iter=1:numSamples
    %sample from old samples depending on transition probs
    randP = unifrnd(0, 1);
    if randP <= transitionR(oldSamples(iter), 1)
        %sample as false '1'
        newSamples(iter) = 1;
    else
        %sample as true '2'
        newSamples(iter) = 2;
    end
    weightSamples(iter) = evidenceU(newSamples(iter), ...
                                    currEvidence);
end

newSamples = weightedSampleWithReplacement(newSamples, ...
                                               weightSamples);

