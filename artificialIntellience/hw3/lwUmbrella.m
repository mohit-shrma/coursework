%use likelihood  weighting to perform approximate inference for
%given sample size, steps and evidence vector
function [pRNumSteps] = lwUmbrella(numSamples, numSteps, evidence)
%{
   CSci5512 Spring'12 Homework 3
   login: sharm163@umn.edu
   date: 4/11/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: lwUmbrella
%}

%using '1' for False and '2' for True
priorR = [0.5; 
          0.5];

%transition prob from R(t-1) to R(t) 
%1,1 means R(t-1) = F to R(t) = T  
transitionR = [0.7 0.3;
              0.3 0.7];

%evidence prob for U(t) given R(t)
%1,1 mean U(t) = F given R(t) = F
evidenceU = [0.8 0.2;
             0.1 0.9];

%add 1 to evidence vec so that 1 denotes false and 2 denotes true
evidence = evidence + 1;

%initialize a vector of samples
samples = zeros(numSamples, 1);

%initialize the above samples vector usin prioR
for iter=1:numSamples
    randP = unifrnd(0, 1);
    %'1' denotes false and '2' denotes true
    if randP <= priorR(1)
        samples(iter) = 1;
    else
        samples(iter) = 2;
    end
end

%initialize a vector of weights of size numSamples
weightSamples = ones(numSamples,1);


%size(nonzeros(samples == 1), 1)
%size(nonzeros(samples == 2), 1)

%for each step of numSteps move the sample one step at a time
for iter=1:numSteps
    %move each sample one step at a time i.e. t-1 to t
    for sampIter=1:numSamples
        [event, weight] = weightedSample(transitionR, evidenceU, ...
                                         samples(sampIter), ...
                                         weightSamples(sampIter), ...
                                         evidence(numSteps));
        samples(sampIter) = event;
        weightSamples(sampIter) = weight;
    end
end


%accumulate weight of 'True' samples
trueWeights = sum(weightSamples(samples == 2));

%accumulate weight of 'False' samples
falseWeights = sum(weightSamples(samples == 1));

%after completion of steps, calculate the probability by counting
pRNumSteps = zeros(2, 1);

%prob that rnumSteps=false
pRNumSteps(1) = falseWeights/(trueWeights + falseWeights);

%prob that rnumSteps=true
pRNumSteps(2) = 1 - pRNumSteps(1);

%considering true var as 1 and false var as 0
%expectation = ((p=t) * 1) + ((p=f) * 0) 
expectation = (pRNumSteps(2) * 1) + (pRNumSteps(1) * 0);

%variance
%variance = (((true var -expectation)**2) * pRNumSteps(2)) + (((false var-expectation)**2)
%* pRNumSteps(2));
variance =  (((1-expectation)^2) * pRNumSteps(2)) + ... 
    (((0-expectation)^2)* pRNumSteps(1));
 
fprintf('\n likelihood weighting, num samples = %d, P(R=F) => %f\n',...
           numSamples, pRNumSteps(1));
 
fprintf('\n likelihood weighting, num samples = %d, P(R=T) => %f\n',...
           numSamples, pRNumSteps(2));
       
fprintf('\n likelihood weighting, num samples =%d, expectation = %f, variance = %f\n',...
          numSamples, expectation, variance);








