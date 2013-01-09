%script for homework 3
%******************* Q1 *****************************%
%{
   CSci5512 Spring'12 Homework 3
   login: sharm163@umn.edu
   date: 4/11/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: hw3Main
%}


%given evidence sequence
evidenceSequence = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

%%fprintf('\nevidence sequence =  ');
evidenceSequence

numSamples = 100;
fprintf('\n******* numSamples = %d *******\n', numSamples);

%probablity by likelihood weighting
pLW = lwUmbrella(numSamples, length(evidenceSequence), evidenceSequence);




%probability by particle filtering
pPF = pfUmbrella(numSamples, length(evidenceSequence), evidenceSequence);

%probability by exact inference filtering
pEf = tpUmbrella(evidenceSequence);



numSamples = 1000;
fprintf('\n******* numSamples = %d *******\n', numSamples);

%probablity by likelihood weighting
pLW = lwUmbrella(numSamples, length(evidenceSequence), evidenceSequence);

%probability by particle filtering
pPF = pfUmbrella(numSamples, length(evidenceSequence), evidenceSequence);

%probability by exact inference filtering
pEf = tpUmbrella(evidenceSequence);


evidenceSequence = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
%%fprintf('\nevidence sequence =  ');
evidenceSequence

fprintf('\n******* numSamples = %d *******\n', numSamples);

numSamples = 100;

%probablity by likelihood weighting
pLW = lwUmbrella(numSamples, length(evidenceSequence), evidenceSequence);

%probability by particle filtering
pPF = pfUmbrella(numSamples, length(evidenceSequence), evidenceSequence);

%probability by exact inference filtering
pEf = tpUmbrella(evidenceSequence);


numSamples = 1000;
fprintf('\n******* numSamples = %d *******\n', numSamples);

%probablity by likelihood weighting
pLW = lwUmbrella(numSamples, length(evidenceSequence), evidenceSequence);

%probability by particle filtering
pPF = pfUmbrella(numSamples, length(evidenceSequence), evidenceSequence);

%probability by exact inference filtering
pEf = tpUmbrella(evidenceSequence);



evidenceSequence = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
%%fprintf('\nevidence sequence =  ');
evidenceSequence

numSamples = 100;
fprintf('\n******* numSamples = %d *******\n', numSamples);

%probablity by likelihood weighting
pLW = lwUmbrella(numSamples, length(evidenceSequence), evidenceSequence);

%probability by particle filtering
pPF = pfUmbrella(numSamples, length(evidenceSequence), evidenceSequence);

%probability by exact inference filtering
pEf = tpUmbrella(evidenceSequence);


numSamples = 1000;
fprintf('\n******* numSamples = %d *******\n', numSamples);

%probablity by likelihood weighting
pLW = lwUmbrella(numSamples, length(evidenceSequence), evidenceSequence);

%probability by particle filtering
pPF = pfUmbrella(numSamples, length(evidenceSequence), evidenceSequence);

%probability by exact inference filtering
pEf = tpUmbrella(evidenceSequence);



%******************* Q3 *****************************%

%value Iteration
fprintf('\n*** value iteration ***\n');
% reward = -2
reward  = -2;
[u,p] = mdpVI(reward);

% reward = -0.2
reward = -0.2;
[u,p] = mdpVI(reward);

% reward = -0.01
reward = -0.01;
[u,p] = mdpVI(reward);

%policy Iteration
fprintf('\n*** policy iteration ***\n');
% reward = -2
reward  = -2;
[u,p] = mdpPI(reward);

% reward = -0.2
reward = -0.2;
[u,p] = mdpPI(reward);

% reward = -0.01
reward = -0.01;
[u,p] = mdpPI(reward);
