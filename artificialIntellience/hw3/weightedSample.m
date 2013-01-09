%returns an event/sample and a weight for passed current evidence value for
%umbrella in rain-umbrella n/w
%transitionR and evidenceU are passed probability matrix and
%evidence given rain matrix
%prevRain is value of R(t-1) '2'/true or '1' /false
function [event, weight] = weightedSample(transitionR, evidenceU, ...
                                  prevRain, prevWeight, currEvidence)
%{
   CSci5512 Spring'12 Homework 3
   login: sharm163@umn.edu
   date: 4/11/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: weightedSample
%}
    
    
%initialize weight to 1
weight = prevWeight;

%get random sample (event) for rain at current step from transitionR given
%prevRain
randP = unifrnd(0, 1);
if randP <= transitionR(prevRain, 1)
    % event is sampled as False 
    event = 1;
else
    % event is sampled as True
    event = 2;
end


%update the weight with sample generated above and
%currEvidence using evidenceU matrix
weight = weight * evidenceU(event, currEvidence);


