function [x] = SmoothHMM(n, inputEvid)
%{
   CSci5512 Spring'12 Homework 2
   login: sharm163@umn.edu
   date: 3/4/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: SmoothHMM
%}

%{
  implementation for smoothing
  e.g. SmoothHMM(10, [0 0 0 1 1 1 1 0 0 0])
%}

%initialize return vector
x = [];

if n ~= length(inputEvid)
    fprintf(['input evidence array dont agree with the length ' ...
             'specified']);
else
    %initial probabilities/ prior
    X0 = [0.5 ; 0.5];

    %transition matrix p(Xt|Xt-1)
    T = [0.7 0.3; 
         0.4 .6];

    %evidence matrix p(Et|Xt)
    E = [0.9 0.1; 
         0.3 0.7];

    %evidence matrix to use given evidence is true 
    evidenceTrue = [0.9 0;
                    0 0.3];

    evidenceFalse = [0.1 0;
                    0 0.7];

    %using 0 for F and 1 for T

    %inputEvid1 = [ 0 0 0 1 1 1 1 0 0 0];
    %inputEvid2 = [ 0 1 0 1 0 1 0 1 0 1];

    %inputEvid = inputEvid1;

    lengthOfEvidence = length(inputEvid);

    smoothedEstimates = ones(2, lengthOfEvidence);

    %compute non norm forward message till last evidence
    %p(x1) = sum<x0>{ p(x1|x0) p(x1) } = T'*X0
    %f11 = p(x1|e1:1) = p(x1|e1) = z p(e1|x1) p(x1) = z E p(x1)
    %nonNormFwd11 = E*T*X0

    %initialize fwdT to prior
    fwdT = X0;
    for iter=1:lengthOfEvidence
        %initialize given evidence matrix to state when evidence given
        %is true
        evidenceMat = evidenceTrue;

        if inputEvid(iter) == 0
            %given evidence is false, change evidenceMat
            evidenceMat = evidenceFalse;
        end

        fwdT = evidenceMat * T' * fwdT;
        fwdT = fwdT / sum(fwdT);
    end

    %compute smooothed estimates running backward algo
    %side by side and using forward message computed above

    %initialize backward message column vector
    bwd = ones(size(X0,1), 1);

    %initialize forward message to last forward message computed above
    fwd = fwdT;
    for iter=1:lengthOfEvidence
        idx =  lengthOfEvidence - iter + 1;
        tempSmoothEstimate = fwd.*bwd;

        %normalize and assign smoothed estimate
        smoothedEstimates(:, idx) = ...
            tempSmoothEstimate/sum(tempSmoothEstimate);

        if idx == 1
            %computed all estimates exit loop
            break;
        end

        %initialize given current evidence matrix to state when evidence given
        %is true
        currEvidenceMat = evidenceTrue;

        if inputEvid(idx) == 0
            %given evidence is false, change evidenceMat
            currEvidenceMat = evidenceFalse;
        end

        %update forward msg for next backward state
        fwd = pinv(T') * pinv(currEvidenceMat) * fwd; 
        fwd = fwd/sum(fwd);

        %initialize previous evidence matrix to true state
        prevEvidenceMat = evidenceTrue;

        if inputEvid(idx-1) == 0
            %given evidence is false, change evidenceMat
            prevEvidenceMat = evidenceFalse;
        end

        %update backward msg for next backward state
        bwd = T * prevEvidenceMat * bwd;

    end

    x = smoothedEstimates;
    x
end

exit;


