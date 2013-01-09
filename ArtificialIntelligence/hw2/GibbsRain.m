function x = GibbsRain(numSteps)
%{
   CSci5512 Spring'12 Homework 2
   login: sharm163@umn.edu
   date: 3/4/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: GibbsRain
%}

% runs gibbs sampling and return the result and input is number of
% steps of iteration.
% e.g: GibbsRain(100000) 

%{
%cpt table convention
%2 => True
%1 => False
%}


%prob C
pCloudy = 0.5;

%probability of S given C
cptSprinklerGivenCloudy = zeros(2);
cptSprinklerGivenCloudy(2) = 0.1;
cptSprinklerGivenCloudy(1) = 0.5;


%probability of R given C
cptRainyGivenCloudy = zeros(2);
cptRainyGivenCloudy(2) = 0.8;
cptRainyGivenCloudy(1) = 0.2;


%probability of wet given S and R
cptWetGivenSprinklerRainy = zeros(2,2);
cptWetGivenSprinklerRainy(2,2) = 0.99;
cptWetGivenSprinklerRainy(2,1) = 0.9;
cptWetGivenSprinklerRainy(1,2) = 0.9;
cptWetGivenSprinklerRainy(1,1) = 0.01;

%probability of C given R & S=T, W=t
cptCloudyGivenRainyNS_T_W_T = zeros(1);

%p(c|r,s,w) = (p(c) p(s|c) p(r|c))/( ( p(c) p(s|c) p(r|c) ) + ( p(~c)
%                                p(s|~c) p(r|~c)))
%p(c|r,s,w) = alpha/(alpha + beta)

alpha = pCloudy * cptSprinklerGivenCloudy(2) * ...
        cptRainyGivenCloudy(2);
beta = (1-pCloudy) * cptSprinklerGivenCloudy(1) * cptRainyGivenCloudy(1);

cptCloudyGivenRainyNS_T_W_T(2) = alpha/(alpha+beta);

%p(c|~r,s,w) = { p(c) p(s|c) p(~r|c)}/ {[ p(c) p(s|c) p(~r|c) ] +
%                                       [p(~c) p(~r|~c) p(s|~c)]}
%p(c|~r,s,w) = alpha/(alpha + beta)

alpha = pCloudy * cptSprinklerGivenCloudy(2) * (1 - ...
                                                cptRainyGivenCloudy(2));
beta = (1-pCloudy) * (1 - cptRainyGivenCloudy(1)) * ...
       cptSprinklerGivenCloudy(1);

cptCloudyGivenRainyNS_T_W_T(1) = alpha / (alpha + beta); 

%probability of R given C & S=T, W=t
cptRainyGivenCloudyNS_T_W_T = zeros(1);

%p(r|c,s,w) = ( p(r|c) p(w|s,r))/( (p(r|c) p(w|s,r) ) + ( p(~r|c)p(w|s,~r)))
%p(r|c,s,w) = alpha/(alpha + beta)

alpha = cptRainyGivenCloudy(2) * cptWetGivenSprinklerRainy(2,2);
beta = (1-cptRainyGivenCloudy(2)) * cptWetGivenSprinklerRainy(2,1);

cptRainyGivenCloudyNS_T_W_T(2) = alpha/(alpha+beta);

%p(r|~c,s,w) = { p(r|~c) p(w|s,r)}/ {[ p(~r|~c) p(w|s,r)] +
%                                       [p(~r|~c) p(w|s,~r)]}
%p(r|~c,s,w) = alpha/(alpha + beta)

alpha = cptRainyGivenCloudy(1) * cptWetGivenSprinklerRainy(2,2);
beta = (1 - cptRainyGivenCloudy(1)) * cptWetGivenSprinklerRainy(2,1);

cptRainyGivenCloudyNS_T_W_T(1) = alpha / (alpha + beta);

%now run the gibbs sampling algo for numSteps iterations

X = GibbsSampler(numSteps, cptRainyGivenCloudyNS_T_W_T, ...
                 cptCloudyGivenRainyNS_T_W_T);

% P(r|s,w)
pRainyGivenS_T_W_T = length(nonzeros(X(:,1) == 2))/size(X,1);
x = pRainyGivenS_T_W_T;
fprintf('\np(R|S=T,W=T) = %d\n', x);
exit;